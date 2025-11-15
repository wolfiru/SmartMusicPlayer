import os
import threading
import time
import json
from io import BytesIO

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, Response, abort

# VLC
try:
    import vlc
    HAVE_VLC = True
except ImportError:
    HAVE_VLC = False
    vlc = None

# Cover-Extractor
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, APIC
    HAVE_MUTAGEN = True
except ImportError:
    HAVE_MUTAGEN = False

# Bildverarbeitung für Cover-Farben
try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

# ==== KONFIGURATION ====
MUSIC_DIR = r"M:\Favoriten (Spotify)"  # fixer Pfad
CSV_PATH = os.path.join(MUSIC_DIR, "song_features_with_clusters.csv")

# presets.json im selben Ordner wie dieses Script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRESET_PATH = os.path.join(BASE_DIR, "presets.json")
# =======================


def tempo_category(bpm):
    if pd.isna(bpm):
        return "unbekannt"
    if bpm < 90:
        return "langsam"
    elif bpm < 120:
        return "mittel"
    else:
        return "schnell"


class WebMusicPlayer:
    """
    Web-gesteuerter Player:
    - Local: VLC spielt am Server
    - Remote: Browser streamt /api/stream/<idx>
    - Intelligente nächste-Titel-Auswahl
    - Presets (Stimmung / Mode) aus presets.json
    - Theme-Farben aus dem Cover
    """

    def __init__(self, music_dir, csv_path):
        self.music_dir = music_dir
        self.csv_path = csv_path

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

        df = pd.read_csv(csv_path)
        if "cluster_id" not in df.columns:
            raise ValueError("CSV hat keine Spalte 'cluster_id'.")

        df["cluster_id"] = df["cluster_id"].astype(int)

        df["tempo_cat"] = df.get("tempo", pd.Series([None] * len(df))).apply(
            tempo_category
        )

        # Energie / Helligkeit-Kategorien
        if "rms_mean" in df.columns:
            q_low, q_high = df["rms_mean"].quantile([0.33, 0.66])
            energy_low, energy_high = q_low, q_high
        else:
            energy_low = energy_high = None

        if "spec_centroid_mean" in df.columns:
            q_low_b, q_high_b = df["spec_centroid_mean"].quantile([0.33, 0.66])
            bright_low, bright_high = q_low_b, q_high_b
        else:
            bright_low = bright_high = None

        def energy_cat(val):
            if pd.isna(val) or energy_low is None:
                return "unbekannt"
            if val < energy_low:
                return "ruhig"
            elif val < energy_high:
                return "normal"
            else:
                return "druckvoll"

        def bright_cat(val):
            if pd.isna(val) or bright_low is None:
                return "unbekannt"
            if val < bright_low:
                return "dunkel"
            elif val < bright_high:
                return "neutral"
            else:
                return "hell"

        df["energy_cat"] = df.get("rms_mean", pd.Series([None] * len(df))).apply(
            energy_cat
        )
        df["bright_cat"] = df.get("spec_centroid_mean", pd.Series([None] * len(df))).apply(
            bright_cat
        )

        self.df = df.reset_index(drop=True)
        self.n_songs = len(self.df)

        # Features für allgemeine Ähnlichkeit
        self.feature_cols = [
            c
            for c in self.df.columns
            if c in ("tempo", "rms_mean", "spec_centroid_mean") or c.startswith("mfcc_")
        ]
        if self.feature_cols:
            feat_mat = self.df[self.feature_cols].copy()
            feat_mat = feat_mat.fillna(feat_mat.mean())
            self.feature_mean = feat_mat.mean(axis=0).values
            self.feature_std = feat_mat.std(axis=0).values
            self.feature_std[self.feature_std == 0] = 1.0
            self.X_norm = (feat_mat.values - self.feature_mean) / self.feature_std

            weights = []
            for col in self.feature_cols:
                if col == "tempo":
                    weights.append(2.0)
                elif col == "rms_mean":
                    weights.append(2.0)
                elif col == "spec_centroid_mean":
                    weights.append(1.0)
                elif col.startswith("mfcc_"):
                    weights.append(0.3)
                else:
                    weights.append(1.0)
            self.weights = np.array(weights)
        else:
            self.X_norm = None
            self.weights = None

        # Features für Presets (Tempo/Energy/Brightness)
        self.preset_cols = ["tempo", "rms_mean", "spec_centroid_mean"]
        if all(c in self.df.columns for c in self.preset_cols):
            preset_mat = self.df[self.preset_cols].copy()
            preset_mat = preset_mat.fillna(preset_mat.mean())
            self.preset_df = preset_mat
            self.preset_min = preset_mat.min()
            self.preset_max = preset_mat.max()
            self.preset_mean = preset_mat.mean()
            self.preset_std = preset_mat.std()
            self.preset_std[self.preset_std == 0] = 1.0
            self.preset_X_norm = (preset_mat - self.preset_mean) / self.preset_std
        else:
            self.preset_df = None
            self.preset_min = None
            self.preset_max = None
            self.preset_mean = None
            self.preset_std = None
            self.preset_X_norm = None

        # Playback-State
        self.current_idx = None
        self.prev_idx = None
        self.played = set()

        # Modus
        self.playback_mode = "local"  # "local" oder "remote"

        # Presets
        self.presets = []
        self.presets_by_id = {}
        self.active_preset_id = None
        self._load_presets()

        # Cover + Theme-Farben
        self.cover_cache = {}         # song_idx -> (mime, data)
        self.theme_color_cache = {}   # song_idx -> "#rrggbb"

        # VLC
        if HAVE_VLC:
            self.vlc_instance = vlc.Instance()
            self.vlc_player = self.vlc_instance.media_player_new()
        else:
            self.vlc_instance = None
            self.vlc_player = None

        self.last_vlc_state = None

        self.lock = threading.Lock()

        # Autoplay-Thread (nur für local relevant)
        t = threading.Thread(target=self._autoadvance_loop, daemon=True)
        t.start()

    # ---------- Presets ----------

    def _load_presets(self):
        if not os.path.exists(PRESET_PATH):
            print(f"[WARN] presets.json nicht gefunden unter {PRESET_PATH}")
            return
        try:
            with open(PRESET_PATH, "r", encoding="utf-8") as f:
                presets = json.load(f)
            self.presets = presets
            self.presets_by_id = {}
            for p in presets:
                pid = int(p.get("id"))
                self.presets_by_id[pid] = p
            print(f"[INFO] {len(self.presets)} Presets geladen.")
        except Exception as e:
            print(f"[ERROR] Konnte presets.json nicht laden: {e}")
            self.presets = []
            self.presets_by_id = {}

    def activate_preset_and_play(self, preset_id):
        """
        Preset aktivieren:
        - Wenn anderer Preset: aktivieren + passendes Lied spielen
        - Wenn gleicher Preset: deaktivieren, kein Autostart
        """
        try:
            preset_id = int(preset_id)
        except (TypeError, ValueError):
            preset_id = None

        with self.lock:
            if preset_id is None or preset_id not in self.presets_by_id:
                self.active_preset_id = None
                return None, False

            if self.active_preset_id == preset_id:
                # toggle off
                self.active_preset_id = None
                print("[INFO] Preset deaktiviert.")
                return None, False

            self.active_preset_id = preset_id
            print(f"[INFO] Preset aktiviert: {self.presets_by_id[preset_id].get('name')} (id={preset_id})")

        # außerhalb des Locks: nächstes Lied starten (Preset-Logik wird in next_song berücksichtigt)
        idx = self.next_song(big_jump=False)
        return idx, True

    # ---------- Hilfen ----------

    def _resolve_path(self, path: str) -> str:
        if not os.path.isabs(path):
            path = os.path.join(self.music_dir, path)
        return path

    def set_mode(self, mode: str):
        with self.lock:
            if mode in ("local", "remote"):
                self.playback_mode = mode
                print(f"[INFO] Playback-Mode geändert auf: {mode}")
            else:
                print(f"[WARN] Ungültiger Mode: {mode}")

    # ---------- Ähnlichkeitslogik ----------

    def _choose_next_index_generic(self, big_jump=False):
        """Standard-Ähnlichkeit (ohne Preset)."""
        if self.X_norm is None or self.weights is None:
            remaining = [i for i in range(self.n_songs) if i not in self.played]
            if not remaining:
                return None
            return int(np.random.choice(remaining))

        all_indices = list(range(self.n_songs))
        remaining = [i for i in all_indices if i not in self.played]
        if not remaining:
            return None

        if self.current_idx is None:
            return int(np.random.choice(remaining))

        v_cur = self.X_norm[self.current_idx]
        X_rem = self.X_norm[remaining]

        diffs = X_rem - v_cur
        d2 = np.sum(self.weights * (diffs ** 2), axis=1)
        d = np.sqrt(d2)

        order = np.argsort(d)
        sorted_idx = [remaining[i] for i in order]
        sorted_d = d[order]
        n = len(sorted_idx)
        if n == 0:
            return None

        if big_jump:
            if n <= 5:
                cand = list(range(n))
            else:
                start = int(0.8 * n)
                cand = list(range(start, n))
        else:
            soft = max(1, int(0.3 * n))
            hard = max(1, int(0.6 * n))
            cand = list(range(soft))
            if not cand:
                cand = list(range(hard))
            if not cand:
                cand = list(range(n))

        cand_d = np.array([sorted_d[i] for i in cand])
        inv = 1.0 / (1.0 + cand_d * cand_d)
        probs = inv / inv.sum() if inv.sum() > 0 else np.ones_like(inv) / len(inv)
        rel_choice = np.random.choice(len(cand), p=probs)
        return int(sorted_idx[cand[rel_choice]])

    def _choose_next_index_preset(self):
        """
        Wählt nächsten Song anhand des aktiven Presets (Tempo/Energie/Brightness).
        Fällt zurück auf Generic, wenn Preset-Features nicht nutzbar sind.
        """
        if self.active_preset_id is None:
            return None

        if self.preset_X_norm is None or self.preset_min is None or self.preset_max is None:
            return None

        preset = self.presets_by_id.get(self.active_preset_id)
        if not preset:
            return None

        # Prozentwerte aus Preset
        p_tempo = float(preset.get("tempo", 50))
        p_energy = float(preset.get("energy", 50))
        p_bright = float(preset.get("brightness", 50))

        def map_percent(col_name, p):
            col_min = self.preset_min[col_name]
            col_max = self.preset_max[col_name]
            if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
                return self.preset_mean[col_name]
            p_clamped = max(0.0, min(100.0, p))
            return col_min + (col_max - col_min) * (p_clamped / 100.0)

        target_vals = {
            "tempo": map_percent("tempo", p_tempo),
            "rms_mean": map_percent("rms_mean", p_energy),
            "spec_centroid_mean": map_percent("spec_centroid_mean", p_bright),
        }

        # in denselben Norm-Raum bringen wie preset_X_norm
        target_norm = []
        for col in self.preset_cols:
            mean = self.preset_mean[col]
            std = self.preset_std[col]
            val = target_vals[col]
            target_norm.append((val - mean) / std)
        target_norm = np.array(target_norm)

        # Kandidaten: zuerst nicht-gespielte, sonst alle
        all_indices = list(range(self.n_songs))
        remaining = [i for i in all_indices if i not in self.played]
        if not remaining:
            remaining = all_indices
            self.played = set()

        X_rem = self.preset_X_norm.iloc[remaining].values
        diffs = X_rem - target_norm
        d2 = np.sum(diffs ** 2, axis=1)  # gleiche Gewichtung für Tempo/Energy/Bright
        d = np.sqrt(d2)

        order = np.argsort(d)
        sorted_idx = [remaining[i] for i in order]
        sorted_d = d[order]
        n = len(sorted_idx)
        if n == 0:
            return None

        # Top-Kandidaten um das Ziel herum
        k = min(30, max(5, n // 3))
        cand = list(range(k))

        cand_d = np.array([sorted_d[i] for i in cand])
        inv = 1.0 / (1.0 + cand_d * cand_d)
        probs = inv / inv.sum() if inv.sum() > 0 else np.ones_like(inv) / len(inv)
        rel_choice = np.random.choice(len(cand), p=probs)
        return int(sorted_idx[cand[rel_choice]])

    def next_song(self, big_jump=False):
        """
        Local: VLC starten
        Remote: nur Index setzen, Browser streamt
        - Wenn Preset aktiv: preset-basierte Auswahl
        - Sonst: generische Ähnlichkeitslogik
        """
        with self.lock:
            attempts = 0
            max_attempts = self.n_songs

            while attempts < max_attempts:
                if self.active_preset_id is not None:
                    idx = self._choose_next_index_preset()
                    if idx is None:
                        # Fallback, falls Preset-Logik nichts findet
                        idx = self._choose_next_index_generic(big_jump=big_jump)
                else:
                    idx = self._choose_next_index_generic(big_jump=big_jump)

                if idx is None:
                    self.current_idx = None
                    return None

                row = self.df.loc[idx]
                path = row["path"]
                full_path = self._resolve_path(path)

                if not os.path.exists(full_path):
                    print(f"[WARN] Datei nicht gefunden: {full_path}")
                    self.played.add(idx)
                    attempts += 1
                    continue

                if self.playback_mode == "local":
                    ok = self._start_vlc(full_path)
                    if ok:
                        self.prev_idx = self.current_idx
                        self.current_idx = idx
                        self.played.add(idx)
                        self.last_vlc_state = None
                        print(f"[INFO] (local) Spiele: {path}")
                        return idx
                    else:
                        print(f"[WARN] (local) Titel übersprungen (VLC-Problem): {path}")
                        self.played.add(idx)
                        attempts += 1
                else:
                    # remote: nur Index setzen
                    self.prev_idx = self.current_idx
                    self.current_idx = idx
                    self.played.add(idx)
                    self.last_vlc_state = None
                    print(f"[INFO] (remote) Neuer Song ausgewählt: {path}")
                    return idx

            print("[ERROR] Keine abspielbaren Titel gefunden.")
        self.current_idx = None
        return None

    # ---------- VLC ----------

    def _start_vlc(self, full_path):
        if not HAVE_VLC or self.vlc_player is None:
            print("[WARN] VLC nicht verfügbar, kann nicht abspielen.")
            return False

        if not os.path.exists(full_path):
            print(f"[WARN] Datei nicht gefunden: {full_path}")
            return False

        try:
            media = self.vlc_instance.media_new(full_path)
            self.vlc_player.set_media(media)
            self.vlc_player.play()
            return True
        except Exception as e:
            print(f"[ERROR] VLC konnte Datei nicht starten: {full_path} ({e})")
            return False

    def pause(self):
        with self.lock:
            if self.playback_mode == "local":
                if HAVE_VLC and self.vlc_player is not None:
                    self.vlc_player.pause()
            # remote: Pause im Browser

    def stop(self):
        with self.lock:
            if HAVE_VLC and self.vlc_player is not None:
                self.vlc_player.stop()
            self.current_idx = None
            self.prev_idx = None
            self.played = set()
            self.last_vlc_state = None
            # Stop setzt auch das Preset zurück
            self.active_preset_id = None

    def _autoadvance_loop(self):
        while True:
            time.sleep(1.0)
            need_next = False

            with self.lock:
                if self.playback_mode != "local":
                    continue

                if not HAVE_VLC or self.vlc_player is None:
                    continue
                if self.current_idx is None:
                    self.last_vlc_state = None
                    continue

                try:
                    state = self.vlc_player.get_state()
                except Exception:
                    continue

                prev_state = self.last_vlc_state
                self.last_vlc_state = state

                if prev_state in (vlc.State.Playing, vlc.State.Paused) and state in (
                    vlc.State.Ended,
                    vlc.State.Stopped,
                    vlc.State.Error,
                ):
                    print(f"[INFO] Autoplay (local): aktueller Song fertig (State={state}), nächster Song...")
                    need_next = True

            if need_next:
                self.next_song(big_jump=False)

    # ---------- Theme-Farbe aus Cover ----------

    def _compute_theme_color(self, data_bytes):
        if not HAVE_PIL:
            return None
        try:
            img = Image.open(BytesIO(data_bytes)).convert("RGB")
            img = img.resize((32, 32))
            arr = np.array(img).reshape(-1, 3)
            mean = arr.mean(axis=0)
            r, g, b = [int(x) for x in mean]
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return None

    # ---------- Status / Ähnlichkeit ----------

    def _compute_similarity(self):
        if (
            self.prev_idx is None
            or self.current_idx is None
            or self.X_norm is None
            or self.weights is None
        ):
            return None, None

        v_prev = self.X_norm[self.prev_idx]
        v_cur = self.X_norm[self.current_idx]
        diff = v_cur - v_prev
        d2 = np.sum(self.weights * (diff ** 2))
        dist = float(np.sqrt(d2))
        sim = 1.0 / (1.0 + dist)
        sim_pct = sim * 100.0
        return sim_pct, dist

    def get_status(self):
        with self.lock:
            if self.current_idx is None:
                return {
                    "mode": self.playback_mode,
                    "playing": False,
                    "vlc_state": "State.Stopped",
                    "current_idx": None,
                    "current_file": None,
                    "cluster_id": None,
                    "tempo": None,
                    "rms": None,
                    "bright": None,
                    "tempo_cat": None,
                    "energy_cat": None,
                    "bright_cat": None,
                    "similarity_percent": None,
                    "similarity_dist": None,
                    "length": None,
                    "position": None,
                    "active_preset_id": self.active_preset_id,
                    "theme_color": None,
                }

            row = self.df.loc[self.current_idx]

            tempo = row.get("tempo", None)
            rms = row.get("rms_mean", None)
            bright = row.get("spec_centroid_mean", None)

            tempo_cat = row.get("tempo_cat", "unbekannt")
            energy_cat = row.get("energy_cat", "unbekannt")
            bright_cat = row.get("bright_cat", "unbekannt")
            cluster_id = int(row.get("cluster_id", -1))

            sim_pct, dist = self._compute_similarity()

            length = None
            position = None
            state = "remote" if self.playback_mode == "remote" else "unknown"
            playing = False

            if self.playback_mode == "local":
                if HAVE_VLC and self.vlc_player is not None:
                    try:
                        length_ms = self.vlc_player.get_length()
                        time_ms = self.vlc_player.get_time()
                        if length_ms > 0:
                            length = length_ms / 1000.0
                        if time_ms >= 0:
                            position = time_ms / 1000.0
                        state = str(self.vlc_player.get_state())
                    except Exception:
                        pass
                playing = state not in ("State.Stopped", "State.Ended")
            else:
                # remote: Browser spielt, Server weiß nur: "sollte laufen"
                playing = True

            theme_color = self.theme_color_cache.get(self.current_idx)

            return {
                "mode": self.playback_mode,
                "playing": playing,
                "vlc_state": state,
                "current_idx": int(self.current_idx),
                "current_file": os.path.basename(row["path"]),
                "cluster_id": cluster_id,
                "tempo": None if pd.isna(tempo) else float(tempo),
                "rms": None if pd.isna(rms) else float(rms),
                "bright": None if pd.isna(bright) else float(bright),
                "tempo_cat": tempo_cat,
                "energy_cat": energy_cat,
                "bright_cat": bright_cat,
                "similarity_percent": None if sim_pct is None else float(sim_pct),
                "similarity_dist": None if dist is None else float(dist),
                "length": length,
                "position": position,
                "active_preset_id": self.active_preset_id,
                "theme_color": theme_color,
            }

    # ---------- Cover ----------

    def get_cover(self, song_idx):
        if not HAVE_MUTAGEN:
            return None, None

        with self.lock:
            if song_idx in self.cover_cache:
                mime, data = self.cover_cache[song_idx]
                return mime, data

        try:
            row = self.df.loc[song_idx]
        except KeyError:
            return None, None

        path = row["path"]
        full_path = self._resolve_path(path)

        if not os.path.exists(full_path):
            return None, None

        # 1) Eingebettetes Cover
        try:
            audio = MP3(full_path, ID3=ID3)
            if audio.tags:
                for tag in audio.tags.values():
                    if isinstance(tag, APIC):
                        mime = tag.mime or "image/jpeg"
                        data = tag.data
                        color = self._compute_theme_color(data)
                        with self.lock:
                            self.cover_cache[song_idx] = (mime, data)
                            if color:
                                self.theme_color_cache[song_idx] = color
                        return mime, data
        except Exception:
            pass

        # 2) Cover-Datei im Ordner (cover/folder.jpg/png)
        folder = os.path.dirname(full_path)
        for name in ("cover.jpg", "cover.png", "folder.jpg", "folder.png"):
            candidate = os.path.join(folder, name)
            if os.path.exists(candidate):
                try:
                    with open(candidate, "rb") as f:
                        data = f.read()
                    mime = "image/jpeg" if candidate.lower().endswith(".jpg") else "image/png"
                    color = self._compute_theme_color(data)
                    with self.lock:
                        self.cover_cache[song_idx] = (mime, data)
                        if color:
                            self.theme_color_cache[song_idx] = color
                    return mime, data
                except Exception:
                    continue

        return None, None

    # ---------- Stream ----------

    def get_song_path(self, song_idx):
        try:
            row = self.df.loc[song_idx]
        except KeyError:
            return None
        path = row["path"]
        full_path = self._resolve_path(path)
        if not os.path.exists(full_path):
            return None
        return full_path


# ================== Flask-App =======================

app = Flask(__name__)
player = WebMusicPlayer(MUSIC_DIR, CSV_PATH)


@app.route("/")
def index():
    html = """
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Smart Music Player</title>
<style>
:root {
  --accent: #8ef;
  --accentText: #000000;
}

html, body {
  height: 100%;
  margin: 0;
  padding: 0;
}
body {
  font-family: sans-serif;
  background:#000;
  color:#eee;
}

/* Hintergrund + Layout */
#bg {
  position: relative;
  min-height: 100vh;
  background:#111;
  background-position: center center;
  background-repeat: no-repeat;
  background-size: cover;
  display:flex;
  flex-direction:column;
}

/* Abdunkelung */
#bg::before {
  content: "";
  position: absolute;
  inset: 0;
  background: rgba(0,0,0,0.8);
  pointer-events: none;
}

#content {
  position: relative;
  z-index: 1;
  padding: 16px;
  flex: 1 1 auto;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;   /* NEU: vertikal zentrieren */
}

/* max-Breite für Desktop, gut lesbar am Handy */
#inner {
  width: 100%;
  max-width: 640px;
}

/* Track-Titel oben */
.track-title {
  font-size: 1.4rem;
  font-weight: 600;
  margin: 0 0 12px 0;
  text-align: center;
  color:#f5f5f5;
  word-wrap: break-word;
}

/* Cover */
#cover {
  width: 80vw;
  max-width: 260px;
  height: auto;
  aspect-ratio: 1 / 1;
  background:#333;
  display:block;
  margin:0 auto 12px auto;
  object-fit:cover;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.6);
}

/* Buttons */
button {
  margin:4px;
  padding:8px 12px;
  font-size:14px;
  border-radius:999px;
  border:none;
  cursor:pointer;
}
button:hover { opacity:0.9; }
.btn-primary { background: var(--accent); color: var(--accentText); }
.btn-secondary { background:#424242; color:#fff; }

.mode-toggle {
  margin-top:6px;
  margin-bottom:6px;
  display:flex;
  justify-content:center;
  flex-wrap:wrap;
}
.btn-toggle {
  margin:4px;
  background:#424242;
  color:#fff;
  padding:6px 14px;
  font-size:13px;
}
.btn-toggle.active {
  background: var(--accent);
  color: var(--accentText);
}

/* Presets */
.preset-container {
  margin-top:8px;
  margin-bottom:8px;
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  justify-content:center;
}
.preset-button {
  background:#333;
  color:#eee;
  border-radius:999px;
  padding:6px 12px;
  font-size:12px;
  border:1px solid #555;
  cursor:pointer;
}
.preset-button.active {
  background: var(--accent);
  color: var(--accentText);
  border-color: var(--accent);
}

/* Info */
.info {
  margin-top:10px;
  font-size: 0.9rem;
}
.info-row {
  display:flex;
  justify-content:space-between;
  gap:10px;
}
.info-row + .info-row {
  margin-top:3px;
}
.label { color:#aaa; }
.value { color:#fff; font-weight:bold; }

/* Progressbar */
.bar {
  width:100%;
  height:8px;
  background:#333;
  margin-top:8px;
  border-radius:4px;
  overflow:hidden;
}
.bar-inner { height:100%; background: var(--accent); width:0%; }
.small { font-size: 0.85em; color:#aaa; margin-top:2px; }

/* Audio-Player */
#audioPlayer {
  width:100%;
  margin-top:4px;
  display:none; /* nur im Remote-Modus sichtbar */
}

/* Footer */
footer {
  position: relative;
  z-index:1;
  flex: 0 0 auto;
  background: rgba(0,0,0,0.75);
  color:#ddd;
  font-size: 0.8rem;
  text-align:center;
  padding:6px 10px;
}

/* Mobile-Touch-Optimierung */
@media (max-width: 480px) {
  .track-title {
    font-size: 1.2rem;
  }
  button {
    font-size:13px;
    padding:7px 10px;
  }
}
</style>
</head>
<body>
<div id="bg">
  <div id="content">
    <div id="inner">
      <h2 id="titleTop" class="track-title">-</h2>

      <img id="cover" src="" alt="Cover">

      <div class="mode-toggle">
        <button id="btnLocal" class="btn-toggle active" onclick="setMode('local')">Lokal (VLC)</button>
        <button id="btnRemote" class="btn-toggle" onclick="setMode('remote')">Remote (Browser)</button>
      </div>

      <audio id="audioPlayer" controls></audio>

      <div id="presets" class="preset-container"></div>

      <div class="info">
        <div class="info-row">
          <span class="label">Cluster:</span>
          <span class="value" id="cluster">-</span>
        </div>
        <div class="info-row">
          <span class="label">Tempo:</span>
          <span class="value">
            <span id="tempo">-</span>
            &nbsp;(<span id="tempo_cat">-</span>)
          </span>
        </div>
        <div class="info-row">
          <span class="label">Energie:</span>
          <span class="value">
            <span id="energy">-</span>
            &nbsp;(<span id="energy_cat">-</span>)
          </span>
        </div>
        <div class="info-row">
          <span class="label">Helligkeit:</span>
          <span class="value">
            <span id="bright">-</span>
            &nbsp;(<span id="bright_cat">-</span>)
          </span>
        </div>
        <div class="info-row">
          <span class="label">Ähnlichkeit:</span>
          <span class="value" id="similarity">-</span>
        </div>
      </div>

      <div class="bar"><div class="bar-inner" id="progress"></div></div>
      <div class="small"><span id="timepos">0:00 / 0:00</span></div>

      <div style="margin-top:12px; text-align:center;">
        <button class="btn-primary" onclick="sendCommand('start')">▶️ Start / Neuer Start</button>
        <button class="btn-primary" onclick="sendCommand('next')">⏭️ Nächster Song</button>
        <button class="btn-secondary" onclick="sendCommand('big_jump')">⏩ Großer Sprung</button>
        <button class="btn-secondary" onclick="sendCommand('pause')">⏯️ Pause/Weiter</button>
        <button class="btn-secondary" onclick="sendCommand('stop')">⏹️ Stop</button>
      </div>
    </div>
  </div>

  <footer>
    Smart Music Player (Raspi / PC)
  </footer>
</div>

<script>
let lastCoverIdx = null;
let playbackMode = 'local';
let lastSongIdx = null;
let presets = [];
let activePresetId = null;

function updateModeUI() {
  const btnLocal = document.getElementById('btnLocal');
  const btnRemote = document.getElementById('btnRemote');
  const audio = document.getElementById('audioPlayer');

  if (playbackMode === 'remote') {
    btnRemote.classList.add('active');
    btnLocal.classList.remove('active');
    audio.style.display = 'block';
  } else {
    btnLocal.classList.add('active');
    btnRemote.classList.remove('active');
    audio.style.display = 'none';
  }
}

function renderPresets() {
  const container = document.getElementById('presets');
  container.innerHTML = '';
  if (!presets || presets.length === 0) {
    const span = document.createElement('span');
    span.textContent = 'Keine Presets geladen.';
    span.style.color = '#777';
    container.appendChild(span);
    return;
  }
  presets.forEach(p => {
    const btn = document.createElement('button');
    btn.className = 'preset-button';
    btn.textContent = p.name;
    btn.onclick = () => selectPreset(p.id);
    if (activePresetId === p.id) {
      btn.classList.add('active');
    }
    container.appendChild(btn);
  });
}

async function loadPresets() {
  try {
    const resp = await fetch('/api/presets');
    const data = await resp.json();
    presets = data.presets || [];
    activePresetId = data.active_id || null;
    renderPresets();
  } catch (e) {
    console.error(e);
  }
}

async function selectPreset(id) {
  try {
    const resp = await fetch('/api/preset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id})
    });
    const data = await resp.json();
    activePresetId = data.active_id;
    presets = data.presets || presets;
    renderPresets();
    setTimeout(fetchStatus, 300);
  } catch (e) {
    console.error(e);
  }
}

async function setMode(mode) {
  playbackMode = mode;
  updateModeUI();
  try {
    await fetch('/api/mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode})
    });
  } catch (e) {
    console.error(e);
  }
}

// Hilfsfunktion: Dateiname → "schöner" Titel
function prettyTitle(filename) {
  if (!filename) return "-";
  let name = filename;

  // Extension weg (.mp3 / .MP3)
  if (name.toLowerCase().endsWith('.mp3')) {
    name = name.slice(0, -4);
  }

  // Führende Tracknummern wie "01 - ", "02_", "03." etc. weg
  name = name.replace(/^\\s*\\d+\\s*[-._]\\s*/, '');

  return name;
}

// Theme-Farbe anwenden
function applyThemeColor(hex) {
  if (!hex) return;
  const m = /^#?([0-9a-fA-F]{6})$/.exec(hex);
  if (!m) return;
  const intVal = parseInt(m[1], 16);
  const r = (intVal >> 16) & 255;
  const g = (intVal >> 8) & 255;
  const b = intVal & 255;
  const lum = (0.2126*r + 0.7152*g + 0.0722*b) / 255;
  const text = lum < 0.4 ? '#ffffff' : '#000000';
  document.documentElement.style.setProperty('--accent', '#' + m[1]);
  document.documentElement.style.setProperty('--accentText', text);
}

async function fetchStatus() {
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();

    if (data.mode && data.mode !== playbackMode) {
      playbackMode = data.mode;
      updateModeUI();
    }

    if ('active_preset_id' in data) {
      activePresetId = data.active_preset_id;
      renderPresets();
    }

    if (data.theme_color) {
      applyThemeColor(data.theme_color);
    }

    if (!data.current_idx && data.current_idx !== 0) {
      document.getElementById('titleTop').textContent = "-";
      document.getElementById('cluster').textContent = "-";
      document.getElementById('tempo').textContent = "-";
      document.getElementById('tempo_cat').textContent = "-";
      document.getElementById('energy').textContent = "-";
      document.getElementById('energy_cat').textContent = "-";
      document.getElementById('bright').textContent = "-";
      document.getElementById('bright_cat').textContent = "-";
      document.getElementById('similarity').textContent = "-";
      document.getElementById('cover').src = "";
      document.getElementById('bg').style.backgroundImage = "none";
      lastCoverIdx = null;
      lastSongIdx = null;
      document.getElementById('progress').style.width = "0%";
      document.getElementById('timepos').textContent = "0:00 / 0:00";
      return;
    }

    // Titel oben
    const niceName = prettyTitle(data.current_file || "");
    document.getElementById('titleTop').textContent = niceName || "-";

    document.getElementById('cluster').textContent = data.cluster_id !== null ? data.cluster_id : "-";

    document.getElementById('tempo').textContent = data.tempo !== null ? data.tempo.toFixed(1) + " BPM" : "-";
    document.getElementById('tempo_cat').textContent = data.tempo_cat || "-";

    document.getElementById('energy').textContent = data.rms !== null ? data.rms.toFixed(4) : "-";
    document.getElementById('energy_cat').textContent = data.energy_cat || "-";

    document.getElementById('bright').textContent = data.bright !== null ? data.bright.toFixed(1) : "-";
    document.getElementById('bright_cat').textContent = data.bright_cat || "-";

    if (data.similarity_percent !== null) {
      document.getElementById('similarity').textContent =
        data.similarity_percent.toFixed(1) + "% (Distanz " + data.similarity_dist.toFixed(3) + ")";
    } else {
      document.getElementById('similarity').textContent = "n/a";
    }

    if (data.current_idx !== null) {
      if (data.current_idx !== lastCoverIdx) {
        const coverUrl = "/api/cover/" + data.current_idx + "?t=" + Date.now();
        document.getElementById('cover').src = coverUrl;
        document.getElementById('bg').style.backgroundImage = "url('" + coverUrl + "')";
        lastCoverIdx = data.current_idx;
      }
    }

    const progressEl = document.getElementById('progress');
    const timeEl = document.getElementById('timepos');
    const audio = document.getElementById('audioPlayer');

    if (playbackMode === 'remote') {
      if (data.current_idx !== null && data.current_idx !== lastSongIdx) {
        const streamUrl = "/api/stream/" + data.current_idx + "?t=" + Date.now();
        audio.src = streamUrl;
        audio.dataset.currentIdx = data.current_idx;
        lastSongIdx = data.current_idx;
        if (data.playing) {
          audio.play().catch(e => console.error(e));
        }
      }

      if (!isNaN(audio.duration) && audio.duration > 0) {
        const pct = Math.max(0, Math.min(100, (audio.currentTime / audio.duration) * 100));
        progressEl.style.width = pct + "%";
        timeEl.textContent = formatTime(audio.currentTime) + " / " + formatTime(audio.duration);
      } else {
        progressEl.style.width = "0%";
        timeEl.textContent = "0:00 / 0:00";
      }
    } else {
      if (data.length && data.position !== null) {
        const pct = Math.max(0, Math.min(100, (data.position / data.length) * 100));
        progressEl.style.width = pct + "%";
        timeEl.textContent =
          formatTime(data.position) + " / " + formatTime(data.length);
      } else {
        progressEl.style.width = "0%";
        timeEl.textContent = "0:00 / 0:00";
      }
    }
  } catch (e) {
    console.error(e);
  }
}

function formatTime(sec) {
  sec = Math.floor(sec || 0);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m + ":" + (s < 10 ? "0" + s : s);
}

async function sendCommand(action) {
  const audio = document.getElementById('audioPlayer');

  if (playbackMode === 'remote' && action === 'pause') {
    if (audio.paused) {
      audio.play().catch(e => console.error(e));
    } else {
      audio.pause();
    }
    return;
  }

  if (playbackMode === 'remote' && action === 'stop') {
    audio.pause();
    audio.currentTime = 0;
  }

  try {
    await fetch('/api/command', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action})
    });
    setTimeout(fetchStatus, 300);
  } catch (e) {
    console.error(e);
  }
}

async function onAudioEnded() {
  if (playbackMode !== 'remote') return;
  try {
    const resp = await fetch('/api/remote_ended', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'}
    });
    const data = await resp.json();
    if (data.idx !== null && data.idx !== undefined) {
      const audio = document.getElementById('audioPlayer');
      const streamUrl = "/api/stream/" + data.idx + "?t=" + Date.now();
      audio.src = streamUrl;
      audio.dataset.currentIdx = data.idx;
      lastSongIdx = data.idx;
      audio.play().catch(e => console.error(e));
    }
  } catch (e) {
    console.error(e);
  }
}

window.addEventListener('load', () => {
  const audio = document.getElementById('audioPlayer');
  audio.addEventListener('ended', onAudioEnded);
  updateModeUI();
  loadPresets();
});

setInterval(fetchStatus, 1000);
fetchStatus();
</script>
</body>
</html>
    """
    return Response(html, mimetype="text/html")


@app.route("/api/status")
def api_status():
    return jsonify(player.get_status())


@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.get_json(force=True) or {}
    action = data.get("action")

    if action == "start":
        player.stop()
        player.next_song(big_jump=False)
    elif action == "next":
        player.next_song(big_jump=False)
    elif action == "big_jump":
        player.next_song(big_jump=True)
    elif action == "pause":
        player.pause()
    elif action == "stop":
        player.stop()

    return jsonify({"ok": True})


@app.route("/api/cover/<int:song_idx>")
def api_cover(song_idx):
    mime, data = player.get_cover(song_idx)
    if mime is None or data is None:
        empty_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10"
            b"\x00\x00\x00\x10\x08\x06\x00\x00\x00\x1f\xf3\xffa"
            b"\x00\x00\x00\x0cIDATx\x9ccddbf\xa0\x040Q\xa4\x8c"
            b"\x82\x81\x81\x01\x00\x0b\x8d\x02\x1d\xea\x14\xc7"
            b"\x9e\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return send_file(BytesIO(empty_png), mimetype="image/png")
    return send_file(BytesIO(data), mimetype=mime)


@app.route("/api/stream/<int:song_idx>")
def api_stream(song_idx):
    full_path = player.get_song_path(song_idx)
    if not full_path:
        abort(404)
    return send_file(full_path, mimetype="audio/mpeg", conditional=True)


@app.route("/api/mode", methods=["GET", "POST"])
def api_mode():
    if request.method == "GET":
        return jsonify({"mode": player.playback_mode})
    data = request.get_json(force=True) or {}
    mode = data.get("mode")
    player.set_mode(mode)
    return jsonify({"ok": True, "mode": player.playback_mode})


@app.route("/api/presets")
def api_presets():
    return jsonify({
        "presets": player.presets,
        "active_id": player.active_preset_id,
    })


@app.route("/api/preset", methods=["POST"])
def api_preset():
    data = request.get_json(force=True) or {}
    preset_id = data.get("id")
    idx, _played = player.activate_preset_and_play(preset_id)
    return jsonify({
        "ok": True,
        "idx": idx,
        "active_id": player.active_preset_id,
        "presets": player.presets,
    })


@app.route("/api/remote_ended", methods=["POST"])
def api_remote_ended():
    idx = player.next_song(big_jump=False)
    status = player.get_status()
    return jsonify({"ok": True, "idx": idx, "status": status})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
