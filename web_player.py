import os
import threading
import time
import json
from io import BytesIO

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, Response, abort, render_template

# VLC
try:
    import vlc
    HAVE_VLC = True
except ImportError:
    HAVE_VLC = False
    vlc = None

# Cover extractor
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, APIC
    HAVE_MUTAGEN = True
except ImportError:
    HAVE_MUTAGEN = False

# Image processing for cover colors
try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

# ==== CONFIGURATION ====
MUSIC_DIR = r"/mnt/musik/Favoriten (Spotify)"  # fixed path
CSV_PATH = os.path.join(MUSIC_DIR, "song_features_with_clusters.csv")

# Default-Output:
#   "Streaming"  -> Browser (remote)
#   "Local"      -> VLC on Server
DEFAULT_OUTPUT = "Streaming"

# presets.json in the same folder as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRESET_PATH = os.path.join(BASE_DIR, "presets.json")
# =======================


def tempo_category(bpm):
    if pd.isna(bpm):
        return "unknown"
    if bpm < 90:
        return "slow"
    elif bpm < 120:
        return "medium"
    else:
        return "fast"


class WebMusicPlayer:
    """
    Web-controlled player:
    - Local: VLC plays on the server
    - Remote: browser streams api/stream/<idx>
    - Intelligent next-track selection
    - Presets (mood / mode) from presets.json
    - Theme colors from the cover
    """

    def __init__(self, music_dir, csv_path):
        self.music_dir = music_dir
        self.csv_path = csv_path

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "cluster_id" not in df.columns:
            raise ValueError("CSV has no column 'cluster_id'.")

        df["cluster_id"] = df["cluster_id"].astype(int)
		
		# Pfade aus dem CSV bereinigen:
        # - Windows-Backslashes in normale Separatoren umwandeln
        # - nur den Dateinamen behalten (M:\Favoriten (Spotify)\... -> CYREES - ...mp3)
        if "path" not in df.columns:
            raise ValueError("CSV has no column 'path'.")

        df["path"] = (
            df["path"]
            .astype(str)
            .apply(lambda p: os.path.basename(p.replace("\\", os.sep)))
        )

        df["tempo_cat"] = df.get("tempo", pd.Series([None] * len(df))).apply(
            tempo_category
        )

        # Energy / brightness categories
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
                return "unknown"
            if val < energy_low:
                return "calm"
            elif val < energy_high:
                return "normal"
            else:
                return "powerful"

        def bright_cat(val):
            if pd.isna(val) or bright_low is None:
                return "unknown"
            if val < bright_low:
                return "dark"
            elif val < bright_high:
                return "neutral"
            else:
                return "bright"

        df["energy_cat"] = df.get("rms_mean", pd.Series([None] * len(df))).apply(
            energy_cat
        )
        df["bright_cat"] = df.get("spec_centroid_mean", pd.Series([None] * len(df))).apply(
            bright_cat
        )

        self.df = df.reset_index(drop=True)
        self.n_songs = len(self.df)

        # Features for general similarity
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

        # Features for presets (tempo/energy/brightness)
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

        # Playback state
        self.current_idx = None
        self.prev_idx = None
        self.played = set()


        # Defaultmode aus Config ableiten
        mode_cfg = (DEFAULT_OUTPUT or "").strip().lower()
        if mode_cfg in ("streaming", "remote", "browser"):
            self.playback_mode = "remote"
        else:
            self.playback_mode = "local"

        print(f"[INFO] Default playback mode: {self.playback_mode}")


        # Presets
        self.presets = []
        self.presets_by_id = {}
        self.active_preset_id = None
        self._load_presets()

        # Cover + theme colors
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

        # Autoplay thread (only relevant for local)
        t = threading.Thread(target=self._autoadvance_loop, daemon=True)
        t.start()

    # ---------- Presets ----------

    def _load_presets(self):
        if not os.path.exists(PRESET_PATH):
            print(f"[WARN] presets.json not found at {PRESET_PATH}")
            return
        try:
            with open(PRESET_PATH, "r", encoding="utf-8") as f:
                presets = json.load(f)
            self.presets = presets
            self.presets_by_id = {}
            for p in presets:
                pid = int(p.get("id"))
                self.presets_by_id[pid] = p
            print(f"[INFO] {len(self.presets)} presets loaded.")
        except Exception as e:
            print(f"[ERROR] Could not load presets.json: {e}")
            self.presets = []
            self.presets_by_id = {}

    def activate_preset_and_play(self, preset_id):
        """
        Activate preset:
        - If different preset: activate + play a matching track
        - If same preset: deactivate, no auto-start
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
                print("[INFO] Preset deactivated.")
                return None, False

            self.active_preset_id = preset_id
            print(
                f"[INFO] Preset activated: "
                f"{self.presets_by_id[preset_id].get('name')} (id={preset_id})"
            )

        # outside the lock: start next track (preset logic is used in next_song)
        idx = self.next_song(big_jump=False)
        return idx, True

    # ---------- Helper ----------

    def _resolve_path(self, path: str) -> str:
        if not os.path.isabs(path):
            path = os.path.join(self.music_dir, path)
        return path

    def set_mode(self, mode: str):
        with self.lock:
            if mode in ("local", "remote"):
                self.playback_mode = mode
                print(f"[INFO] Playback mode changed to: {mode}")
            else:
                print(f"[WARN] Invalid mode: {mode}")

    # ---------- Similarity logic ----------

    def _choose_next_index_generic(self, big_jump=False):
        if self.X_norm is None or self.weights is None:
            remaining = [i for i in range(self.n_songs) if i not in self.played]
            if not remaining:
                return None
            return int(np.random.choice(remaining))

        all_indices = list(range(self.n_songs))
        remaining = [i for i in all_indices if i not in self.played]
        if not remaining:
            return None

        # Erster Track: komplett zufällig aus allen verbleibenden
        if self.current_idx is None:
            return int(np.random.choice(remaining))

        # Ab dem zweiten Track: mit einer gewissen Wahrscheinlichkeit komplett zufällig
        # -> bringt mehr Durchmischung, statt immer nur "stark ähnliche" Songs.
        if np.random.rand() < 0.2:  # 20% reine "Exploration"
            return int(np.random.choice(remaining))

        # Bisherige Similarity-Logik
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
        Choose next track based on active preset (tempo/energy/brightness).
        Falls back to generic similarity if preset features are not usable.
        """
        if self.active_preset_id is None:
            return None

        if (
            self.preset_X_norm is None
            or self.preset_min is None
            or self.preset_max is None
        ):
            return None

        preset = self.presets_by_id.get(self.active_preset_id)
        if not preset:
            return None

        # Percent values from preset
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

        # transform to same normalized space as preset_X_norm
        target_norm = []
        for col in self.preset_cols:
            mean = self.preset_mean[col]
            std = self.preset_std[col]
            val = target_vals[col]
            target_norm.append((val - mean) / std)
        target_norm = np.array(target_norm)

        # candidates: first non-played, otherwise all
        all_indices = list(range(self.n_songs))
        remaining = [i for i in all_indices if i not in self.played]
        if not remaining:
            remaining = all_indices
            self.played = set()

        X_rem = self.preset_X_norm.iloc[remaining].values
        diffs = X_rem - target_norm
        d2 = np.sum(diffs ** 2, axis=1)  # equal weighting for tempo/energy/brightness
        d = np.sqrt(d2)

        order = np.argsort(d)
        sorted_idx = [remaining[i] for i in order]
        sorted_d = d[order]
        n = len(sorted_idx)
        if n == 0:
            return None

        # top candidates around the target
        k = min(30, max(5, n // 3))
        cand = list(range(k))

        cand_d = np.array([sorted_d[i] for i in cand])
        inv = 1.0 / (1.0 + cand_d * cand_d)
        probs = inv / inv.sum() if inv.sum() > 0 else np.ones_like(inv) / len(inv)
        rel_choice = np.random.choice(len(cand), p=probs)
        return int(sorted_idx[cand[rel_choice]])

    def next_song(self, big_jump=False):
        """
        Local: start VLC
        Remote: only set index, browser streams
        - If preset active: preset-based selection
        - Otherwise: generic similarity logic
        """
        with self.lock:
            attempts = 0
            max_attempts = self.n_songs

            while attempts < max_attempts:
                if self.active_preset_id is not None:
                    idx = self._choose_next_index_preset()
                    if idx is None:
                        # fallback if preset logic finds nothing
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
                    print(f"[WARN] File not found: {full_path}")
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
                        print(f"[INFO] (local) Playing: {path}")
                        return idx
                    else:
                        print(
                            f"[WARN] (local) Skipped track (VLC issue): {path}"
                        )
                        self.played.add(idx)
                        attempts += 1
                else:
                    # remote: only set index
                    self.prev_idx = self.current_idx
                    self.current_idx = idx
                    self.played.add(idx)
                    self.last_vlc_state = None
                    print(f"[INFO] (remote) Selected new track: {path}")
                    return idx

            print("[ERROR] No playable tracks found.")
        self.current_idx = None
        return None

    # ---------- VLC ----------

    def _start_vlc(self, full_path):
        if not HAVE_VLC or self.vlc_player is None:
            print("[WARN] VLC not available, cannot play.")
            return False

        if not os.path.exists(full_path):
            print(f"[WARN] File not found: {full_path}")
            return False

        try:
            media = self.vlc_instance.media_new(full_path)
            self.vlc_player.set_media(media)
            self.vlc_player.play()
            return True
        except Exception as e:
            print(f"[ERROR] VLC could not start file: {full_path} ({e})")
            return False

    def pause(self):
        with self.lock:
            if self.playback_mode == "local":
                if HAVE_VLC and self.vlc_player is not None:
                    self.vlc_player.pause()
            # remote: pause handled in the browser

    def stop(self):
        with self.lock:
            if HAVE_VLC and self.vlc_player is not None:
                self.vlc_player.stop()
            self.current_idx = None
            self.prev_idx = None
            self.played = set()
            self.last_vlc_state = None
            # stop also resets the preset
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
                    print(
                        f"[INFO] Autoplay (local): track finished (state={state}), next track..."
                    )
                    need_next = True

            if need_next:
                self.next_song(big_jump=False)

    # ---------- Theme color from cover ----------

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

    # ---------- Status / similarity ----------

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

            tempo_cat = row.get("tempo_cat", "unknown")
            energy_cat = row.get("energy_cat", "unknown")
            bright_cat = row.get("bright_cat", "unknown")
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
                # remote: browser plays, server only knows "should be running"
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

        # 1) embedded cover
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

        # 2) cover file in folder (cover/folder.jpg/png)
        folder = os.path.dirname(full_path)
        for name in ("cover.jpg", "cover.png", "folder.jpg", "folder.png"):
            candidate = os.path.join(folder, name)
            if os.path.exists(candidate):
                try:
                    with open(candidate, "rb") as f:
                        data = f.read()
                    mime = (
                        "image/jpeg"
                        if candidate.lower().endswith(".jpg")
                        else "image/png"
                    )
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


# ================== Flask app =======================

app = Flask(__name__)
player = WebMusicPlayer(MUSIC_DIR, CSV_PATH)


@app.route("/")
def index():
    return render_template("player.html")




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
