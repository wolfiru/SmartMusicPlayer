import os
import random
import time
import sys

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.traceback import install as rich_traceback_install
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

# Schönere Tracebacks
rich_traceback_install()
console = Console()

# VLC optional laden
try:
    import vlc

    HAVE_VLC = True
except ImportError:
    HAVE_VLC = False
    vlc = None

# Unter Windows: non-blocking Tastaturabfrage
ON_WINDOWS = sys.platform.startswith("win")
if ON_WINDOWS:
    try:
        import msvcrt
    except ImportError:
        ON_WINDOWS = False

# === KONFIGURATION ===
MUSIC_DIR = r"M:\Favoriten (Spotify)"  # <-- anpassen wie beim Analyzer
CSV_PATH = os.path.join(MUSIC_DIR, "song_features_with_clusters.csv")


def tempo_category(bpm):
    if pd.isna(bpm):
        return "unbekannt"
    if bpm < 90:
        return "langsam"
    elif bpm < 120:
        return "mittel"
    else:
        return "schnell"


class SmartClusterPlayer:
    def __init__(self, music_dir, csv_path):
        self.music_dir = music_dir
        self.csv_path = csv_path

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                "CSV nicht gefunden: %s\nBitte zuerst den Analyzer laufen lassen." % csv_path
            )

        console.rule("[bold cyan]SmartClusterPlayer[/bold cyan]")
        console.print("Lade Cluster- und Feature-Daten aus:\n[bold]%s[/bold]" % csv_path)

        self.df = pd.read_csv(csv_path)
        if "cluster_id" not in self.df.columns:
            raise ValueError("CSV hat keine Spalte 'cluster_id'.")

        self.df["cluster_id"] = self.df["cluster_id"].astype(int)

        # Tempo-Kategorie je Song
        self.df["tempo_cat"] = self.df.get("tempo", pd.Series([None] * len(self.df))).apply(
            tempo_category
        )

        # Globale Schwellen für Energie und Helligkeit
        if "rms_mean" in self.df.columns:
            q_low, q_high = self.df["rms_mean"].quantile([0.33, 0.66])
            self.energy_low = q_low
            self.energy_high = q_high
        else:
            self.energy_low = None
            self.energy_high = None

        if "spec_centroid_mean" in self.df.columns:
            q_low_b, q_high_b = self.df["spec_centroid_mean"].quantile([0.33, 0.66])
            self.bright_low = q_low_b
            self.bright_high = q_high_b
        else:
            self.bright_low = None
            self.bright_high = None

        def energy_category(val):
            if pd.isna(val) or self.energy_low is None:
                return "unbekannt"
            if val < self.energy_low:
                return "ruhig"
            elif val < self.energy_high:
                return "normal"
            else:
                return "druckvoll"

        def brightness_category(val):
            if pd.isna(val) or self.bright_low is None:
                return "unbekannt"
            if val < self.bright_low:
                return "dunkel"
            elif val < self.bright_high:
                return "neutral"
            else:
                return "hell"

        self.df["energy_cat"] = self.df.get("rms_mean", pd.Series([None] * len(self.df))).apply(
            energy_category
        )
        self.df["bright_cat"] = self.df.get(
            "spec_centroid_mean", pd.Series([None] * len(self.df))
        ).apply(brightness_category)

        self.groups = self.df.groupby("cluster_id")
        self.all_cluster_ids = sorted(self.groups.groups.keys())

        # Session-Zustand Cluster-Modus
        self.current_cluster_id = None
        self.played_in_cluster = set()
        self.blocked_clusters = set()

        # Modus: 'cluster' oder 'similar'
        self.mode = "cluster"

        # Ähnlichkeitsmodus-Features
        self.feature_cols = [
            c
            for c in self.df.columns
            if c in ("tempo", "rms_mean", "spec_centroid_mean") or c.startswith("mfcc_")
        ]
        if not self.feature_cols:
            console.print(
                "[yellow]Warnung: keine geeigneten Feature-Spalten für Ähnlichkeitsmodus gefunden.[/yellow]"
            )
            self.X_norm = None
            self.weights = None
        else:
            feat_mat = self.df[self.feature_cols].copy()
            feat_mat = feat_mat.fillna(feat_mat.mean())
            self.feature_mean = feat_mat.mean(axis=0).values
            self.feature_std = feat_mat.std(axis=0).values
            self.feature_std[self.feature_std == 0] = 1.0
            self.X_norm = (feat_mat.values - self.feature_mean) / self.feature_std

            # Gewichte: Tempo/Energie stark, Helligkeit mittel, MFCCs leicht
            w = []
            for col in self.feature_cols:
                if col == "tempo":
                    w.append(2.0)
                elif col == "rms_mean":
                    w.append(2.0)
                elif col == "spec_centroid_mean":
                    w.append(1.0)
                elif col.startswith("mfcc_"):
                    w.append(0.3)
                else:
                    w.append(1.0)
            self.weights = np.array(w)

        # Ähnlichkeitsmodus: aktueller Index & gespielte Songs
        self.sim_current_idx = None
        self.sim_played = set()
        self.sim_force_big_jump = False

        # Index-Tracking für Ähnlichkeitsanzeige
        self.current_idx = None
        self.prev_idx = None

        # VLC-Player
        if HAVE_VLC:
            self.vlc_instance = vlc.Instance()
        else:
            self.vlc_instance = None
        self.vlc_player = None

    # ---------- Modus-Auswahl ----------

    def select_mode(self):
        console.print(
            "\n[bold]Modus-Auswahl:[/bold]\n"
            "[b]1[/b] → Cluster-Modus (nach Gruppen/Moods)\n"
            "[b]2[/b] → ÄHNLICHKEITS-Modus (intelligente Playlist, kleine Sprünge)\n"
        )
        while True:
            choice = Prompt.ask("Modus (1/2)", default="1").strip()
            if choice == "1":
                self.mode = "cluster"
                console.print("[green]Cluster-Modus gewählt.[/green]")
                self.select_initial_cluster()
                return
            elif choice == "2":
                if self.X_norm is None or self.weights is None:
                    console.print(
                        "[red]Ähnlichkeitsmodus nicht verfügbar (keine Features vorhanden).[/red]"
                    )
                    continue
                self.mode = "similar"
                console.print("[green]Ähnlichkeitsmodus gewählt.[/green]")
                self.init_similar_mode()
                return
            else:
                console.print("[red]Bitte 1 oder 2 eingeben.[/red]")

    # ---------- Cluster-Infos / Stats ----------

    def get_cluster_stats(self):
        agg = self.df.groupby("cluster_id").agg(
            anzahl_songs=("path", "count"),
            avg_tempo=("tempo", "mean"),
            avg_rms=("rms_mean", "mean"),
            avg_bright=("spec_centroid_mean", "mean"),
        ).reset_index()

        def mode_for(col, cid):
            series = self.df[self.df["cluster_id"] == cid][col].dropna()
            if series.empty:
                return "unbekannt"
            return series.mode().iloc[0]

        agg["tempo_cat"] = agg["cluster_id"].apply(lambda cid: mode_for("tempo_cat", cid))
        agg["energy_cat"] = agg["cluster_id"].apply(lambda cid: mode_for("energy_cat", cid))
        agg["bright_cat"] = agg["cluster_id"].apply(lambda cid: mode_for("bright_cat", cid))

        return agg.sort_values("cluster_id")

    def show_cluster_overview(self):
        stats = self.get_cluster_stats()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Cluster-ID", justify="center")
        table.add_column("Anz.", justify="right")
        table.add_column("Ø BPM", justify="right")
        table.add_column("Tempo", justify="center")
        table.add_column("Energie", justify="center")
        table.add_column("Helligkeit", justify="center")

        for _, row in stats.iterrows():
            bpm_str = "-" if pd.isna(row["avg_tempo"]) else "%.1f" % row["avg_tempo"]
            table.add_row(
                str(int(row["cluster_id"])),
                str(int(row["anzahl_songs"])),
                bpm_str,
                row["tempo_cat"],
                row["energy_cat"],
                row["bright_cat"],
            )

        console.print("\n[bold cyan]Verfügbare Cluster:[/bold cyan]")
        console.print(table)

    # ---------- Cluster-Modus: Auswahl ----------

    def pick_random_cluster(self):
        candidates = [cid for cid in self.all_cluster_ids if cid not in self.blocked_clusters]
        if not candidates:
            console.print("[red]Es sind keine Cluster mehr frei (alle geblockt).[/red]")
            return None
        return random.choice(candidates)

    def pick_random_cluster_by_tempo(self, tempo_cat):
        stats = self.get_cluster_stats()
        candidates = stats[
            (stats["tempo_cat"] == tempo_cat)
            & (~stats["cluster_id"].isin(self.blocked_clusters))
        ]["cluster_id"].tolist()
        if not candidates:
            return None
        return random.choice(candidates)

    def select_by_filters(self):
        stats = self.get_cluster_stats()

        console.print("\n[bold cyan]Filter-Auswahl[/bold cyan]")
        tempo_opt = Prompt.ask(
            "Tempo (l=langsam, m=mittel, s=schnell, *=egal)", default="*"
        ).lower()
        energy_opt = Prompt.ask(
            "Energie (r=ruhig, n=normal, d=druckvoll, *=egal)", default="*"
        ).lower()
        bright_opt = Prompt.ask(
            "Helligkeit (d=dunkel, n=neutral, h=hell, *=egal)", default="*"
        ).lower()

        df = stats.copy()

        if tempo_opt in ("l", "m", "s"):
            t_map = {"l": "langsam", "m": "mittel", "s": "schnell"}
            df = df[df["tempo_cat"] == t_map[tempo_opt]]

        if energy_opt in ("r", "n", "d"):
            e_map = {"r": "ruhig", "n": "normal", "d": "druckvoll"}
            df = df[df["energy_cat"] == e_map[energy_opt]]

        if bright_opt in ("d", "n", "h"):
            b_map = {"d": "dunkel", "n": "neutral", "h": "hell"}
            df = df[df["bright_cat"] == b_map[bright_opt]]

        df = df[~df["cluster_id"].isin(self.blocked_clusters)]

        if df.empty:
            console.print("[yellow]Kein Cluster passt zu diesen Filtern.[/yellow]")
            return

        cid = random.choice(df["cluster_id"].tolist())
        self.set_cluster(int(cid))

    def select_initial_cluster(self):
        self.show_cluster_overview()

        console.print(
            "\n[bold]Cluster-Auswahl:[/bold]\n"
            "[b]ID[/b]   → konkreten Cluster (z.B. 3)\n"
            "[b]r[/b]    → zufälliger Cluster\n"
            "[b]l[/b]    → langsamer Cluster\n"
            "[b]m[/b]    → mittlerer Cluster\n"
            "[b]s[/b]    → schneller Cluster\n"
            "[b]f[/b]    → Filter nach Tempo/Energie/Helligkeit\n"
        )

        while True:
            choice = Prompt.ask("Auswahl (ID/r/l/m/s/f)").strip().lower()

            if choice == "r":
                cid = self.pick_random_cluster()
                if cid is not None:
                    self.set_cluster(cid)
                    return
            elif choice in ("l", "m", "s"):
                tempo_map = {"l": "langsam", "m": "mittel", "s": "schnell"}
                cid = self.pick_random_cluster_by_tempo(tempo_map[choice])
                if cid is not None:
                    self.set_cluster(cid)
                    return
                else:
                    console.print(
                        "[yellow]Kein Cluster mit Tempo '%s' gefunden.[/yellow]"
                        % tempo_map[choice]
                    )
            elif choice == "f":
                self.select_by_filters()
                if self.current_cluster_id is not None:
                    return
            else:
                try:
                    cid = int(choice)
                    if cid in self.all_cluster_ids:
                        self.set_cluster(cid)
                        return
                    else:
                        console.print("[red]Ungültige Cluster-ID.[/red]")
                except ValueError:
                    console.print("[red]Eingabe nicht verstanden.[/red]")

    def set_cluster(self, cluster_id):
        self.current_cluster_id = cluster_id
        self.played_in_cluster = set()

        cluster_df = self.groups.get_group(cluster_id)
        tempo_vals = cluster_df.get("tempo")
        avg_bpm = tempo_vals.mean() if tempo_vals is not None else None
        cat = tempo_category(avg_bpm) if avg_bpm is not None else "unbekannt"

        console.print(
            "\n[bold green]Starte mit Cluster %d[/bold green]" % cluster_id
        )
        if avg_bpm is not None:
            console.print(
                "Songs: [bold]%d[/bold], Ø Tempo: [bold]%.1f[/bold] BPM (%s)"
                % (len(cluster_df), avg_bpm, cat)
            )
        else:
            console.print("Songs: [bold]%d[/bold]" % len(cluster_df))

    # ---------- Cluster-Modus: nächste Songs ----------

    def get_next_song_in_cluster(self):
        if self.current_cluster_id is None:
            self.select_initial_cluster()

        cluster_df = self.groups.get_group(self.current_cluster_id)
        candidates = cluster_df[~cluster_df.index.isin(self.played_in_cluster)]

        if candidates.empty:
            console.print(
                "[yellow]Cluster %d ist für diese Session leer.[/yellow]"
                % self.current_cluster_id
            )
            self.blocked_clusters.add(self.current_cluster_id)
            new_cid = self.pick_random_cluster()
            if new_cid is None:
                return None, None
            self.set_cluster(new_cid)
            cluster_df = self.groups.get_group(self.current_cluster_id)
            candidates = cluster_df

        row = candidates.sample(1).iloc[0]
        idx = row.name
        self.played_in_cluster.add(idx)

        # Index-Tracking für Ähnlichkeitsanzeige
        self.prev_idx = self.current_idx
        self.current_idx = idx

        return row["path"], idx

    # ---------- Ähnlichkeitsmodus: Initialisierung & Logik ----------

    def init_similar_mode(self):
        """Startsong für Ähnlichkeitsmodus wählen."""
        self.sim_played = set()
        self.sim_current_idx = None
        self.prev_idx = None
        self.current_idx = None

        console.print(
            "\n[bold cyan]Start im Ähnlichkeitsmodus[/bold cyan]\n"
            "[b]r[/b] → zufälliger Startsong\n"
            "[b]c[/b] → Startsong aus einem gewählten Cluster\n"
        )
        choice = Prompt.ask("Auswahl (r/c)", default="r").strip().lower()

        if choice == "c":
            # Cluster wählen und zufälligen Song daraus nehmen
            self.select_initial_cluster()
            cluster_df = self.groups.get_group(self.current_cluster_id)
            row = cluster_df.sample(1).iloc[0]
        else:
            # komplett zufälliger Song
            row = self.df.sample(1).iloc[0]

        idx = row.name
        self.sim_current_idx = idx
        self.sim_played.add(idx)

        # Index-Tracking
        self.prev_idx = None
        self.current_idx = idx

        console.print(
            "[green]Ähnlichkeitsmodus gestartet mit:[/green] %s"
            % os.path.basename(row["path"])
        )

    def _choose_next_similar_index(self, big_jump=False):
        """Wähle den nächsten Song-Index basierend auf Distanz im Feature-Raum."""
        if self.X_norm is None or self.weights is None:
            return None

        all_indices = list(range(len(self.df)))
        remaining = [i for i in all_indices if i not in self.sim_played]

        if not remaining:
            return None

        v_cur = self.X_norm[self.sim_current_idx]
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
            # bewusster Stil-Sprung: nimm aus dem „entfernten“ Teil (oberste 20%)
            if n <= 5:
                cand_indices = list(range(n))
            else:
                start = int(0.8 * n)
                cand_indices = list(range(start, n))
        else:
            # normaler Modus: erst sehr ähnliche, dann moderat andere
            soft = max(1, int(0.3 * n))
            hard = max(1, int(0.6 * n))

            cand_indices = list(range(soft))
            if not cand_indices:
                cand_indices = list(range(hard))
            if not cand_indices:
                cand_indices = list(range(n))

        cand_d = np.array([sorted_d[i] for i in cand_indices])
        inv = 1.0 / (1.0 + cand_d * cand_d)
        inv_sum = inv.sum()
        if inv_sum <= 0:
            probs = np.ones_like(inv) / len(inv)
        else:
            probs = inv / inv_sum

        rel_choice = np.random.choice(len(cand_indices), p=probs)
        chosen_global = sorted_idx[cand_indices[rel_choice]]
        return chosen_global

    def get_next_song_similar(self):
        """Nächster Song im Ähnlichkeitsmodus."""
        if self.sim_current_idx is None:
            self.init_similar_mode()
            idx = self.sim_current_idx
            path = self.df.loc[idx, "path"]
            return path, idx

        next_idx = self._choose_next_similar_index(big_jump=self.sim_force_big_jump)
        self.sim_force_big_jump = False

        if next_idx is None:
            return None, None

        self.sim_current_idx = next_idx
        self.sim_played.add(next_idx)

        # Index-Tracking
        self.prev_idx = self.current_idx
        self.current_idx = next_idx

        return self.df.loc[next_idx, "path"], next_idx

    def skip_group_similar(self):
        """Im Ähnlichkeitsmodus: erzwinge beim nächsten Song einen großen Stil-Sprung."""
        console.print("[yellow]Großer Sprung im Ähnlichkeitsmodus angefordert.[/yellow]")
        self.sim_force_big_jump = True

    # ---------- Audio ----------

    def stop_audio(self):
        if HAVE_VLC and self.vlc_player is not None:
            self.vlc_player.stop()

    def start_audio(self, path):
        if not HAVE_VLC:
            console.print(
                "[yellow]VLC-Playback nicht verfügbar (python-vlc nicht installiert).[/yellow]\n"
                "[yellow]Würde spielen:[/yellow] %s" % path
            )
            return

        if self.vlc_player is not None:
            self.vlc_player.stop()

        media = self.vlc_instance.media_new(path)
        self.vlc_player = self.vlc_instance.media_player_new()
        self.vlc_player.set_media(media)
        self.vlc_player.play()

    def _print_song_info(self, song_path):
        """Zusätzliche Parameter + Ähnlichkeitsfaktor ausgeben."""
        filename = os.path.basename(song_path)
        if self.mode == "cluster":
            mode_tag = f"Cluster {self.current_cluster_id}"
        else:
            mode_tag = "Ähnlichkeitsmodus"

        console.print(
            "\n[bold green]Jetzt läuft:[/bold green] %s (%s)"
            % (filename, mode_tag)
        )

        # Song-Parameter
        if self.current_idx is None or self.current_idx not in self.df.index:
            return

        row = self.df.loc[self.current_idx]

        tempo = row.get("tempo", float("nan"))
        rms = row.get("rms_mean", float("nan"))
        bright = row.get("spec_centroid_mean", float("nan"))

        tempo_str = "-" if pd.isna(tempo) else f"{tempo:.1f} BPM"
        rms_str = "-" if pd.isna(rms) else f"{rms:.4f}"
        bright_str = "-" if pd.isna(bright) else f"{bright:.1f}"

        tempo_cat = row.get("tempo_cat", "unbekannt")
        energy_cat = row.get("energy_cat", "unbekannt")
        bright_cat = row.get("bright_cat", "unbekannt")

        cid = row.get("cluster_id", None)
        if pd.isna(cid):
            cluster_str = "Cluster: -"
        else:
            cluster_str = f"Cluster: {int(cid)}"

        console.print(
            f"[cyan]Info:[/cyan] Tempo: {tempo_str} | Energie (RMS): {rms_str} | Helligkeit: {bright_str}"
        )
        console.print(
            f"[cyan]Mood:[/cyan] Tempo={tempo_cat}, Energie={energy_cat}, Helligkeit={bright_cat} | {cluster_str}"
        )

        # Ähnlichkeitsfaktor
        if (
            self.prev_idx is None
            or self.X_norm is None
            or self.weights is None
            or self.prev_idx < 0
            or self.prev_idx >= len(self.df)
            or self.current_idx < 0
            or self.current_idx >= len(self.df)
        ):
            console.print("[cyan]Ähnlichkeit zum vorherigen Lied:[/cyan] n/a (erstes Lied oder keine Daten)")
        else:
            v_prev = self.X_norm[self.prev_idx]
            v_cur = self.X_norm[self.current_idx]
            diff = v_cur - v_prev
            d2 = np.sum(self.weights * (diff ** 2))
            dist = float(np.sqrt(d2))
            sim = 1.0 / (1.0 + dist)
            sim_pct = sim * 100.0
            console.print(
                f"[cyan]Ähnlichkeit zum vorherigen Lied:[/cyan] {sim_pct:.1f}% (Distanz: {dist:.3f})"
            )

    def play_song_with_progress(self, song_path):
        """
        Spielt einen Song, zeigt Fortschritt, hört auf Tastendrücke.
        Rückgabe: Aktion als String:
          'finished', 'next', 'skip_group', 'change_cluster', 'quit'
        """
        # Songinfo (inkl. Ähnlichkeit) ausgeben
        self._print_song_info(song_path)

        if not HAVE_VLC:
            choice = Prompt.ask(
                "[Enter]=weiter, s=Song skip, g=Gruppe/Big Jump, c=Neu wählen, q=quit",
                default="",
            ).strip().lower()
            if choice == "q":
                return "quit"
            elif choice == "g":
                return "skip_group"
            elif choice == "c":
                return "change_cluster"
            else:
                return "next"

        # Mit VLC: Fortschritt anzeigen
        self.start_audio(song_path)

        # etwas warten, bis VLC die Länge kennt
        length_ms = -1
        for _ in range(25):  # ~5 Sekunden
            length_ms = self.vlc_player.get_length()
            if length_ms > 0:
                break
            time.sleep(0.2)

        if length_ms <= 0:
            total_seconds = None
        else:
            total_seconds = length_ms / 1000.0

        console.print(
            "[cyan]Steuerung:[/cyan] Enter=sofort nächster Song • s=Song skip • "
            "g=Gruppe/Big Jump • c=Neu wählen • q=quit"
        )

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "verbleibend",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            if total_seconds is not None:
                task = progress.add_task(
                    "Spiele: %s" % os.path.basename(song_path)[:30],
                    total=total_seconds,
                )
            else:
                task = progress.add_task(
                    "Spiele: %s" % os.path.basename(song_path)[:30],
                    total=100,
                )

            while True:
                state = self.vlc_player.get_state()

                if state in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
                    return "finished"

                if total_seconds is not None:
                    cur_ms = self.vlc_player.get_time()
                    if cur_ms >= 0:
                        progress.update(task, completed=cur_ms / 1000.0)
                else:
                    if progress.tasks[task].completed < 100:
                        progress.advance(task, 0.2)

                if ON_WINDOWS and msvcrt is not None and msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch == "\r":  # Enter
                        return "next"
                    ch_lower = ch.lower()
                    if ch_lower == "q":
                        return "quit"
                    elif ch_lower == "s":
                        return "next"
                    elif ch_lower == "g":
                        return "skip_group"
                    elif ch_lower == "c":
                        return "change_cluster"

                time.sleep(0.2)

    # ---------- Gruppe wechseln ----------

    def skip_group_cluster(self):
        if self.current_cluster_id is not None:
            console.print(
                "[yellow]Gruppe %d wird übersprungen.[/yellow]" % self.current_cluster_id
            )
            self.blocked_clusters.add(self.current_cluster_id)
        new_cid = self.pick_random_cluster()
        if new_cid is None:
            return None
        self.set_cluster(new_cid)
        return self.get_next_song_in_cluster()


def main():
    player = SmartClusterPlayer(MUSIC_DIR, CSV_PATH)
    player.select_mode()

    while True:
        if player.mode == "cluster":
            song_path, idx = player.get_next_song_in_cluster()
        else:
            song_path, idx = player.get_next_song_similar()

        if song_path is None:
            console.print("[red]Keine Songs mehr verfügbar. Session endet.[/red]")
            break

        action = player.play_song_with_progress(song_path)

        if action == "quit":
            player.stop_audio()
            console.print("[bold]Beende Player.[/bold]")
            break
        elif action == "skip_group":
            player.stop_audio()
            if player.mode == "cluster":
                next_song = player.skip_group_cluster()
                if next_song is None:
                    console.print("[red]Keine neue Gruppe verfügbar.[/red]")
                    break
            else:
                player.skip_group_similar()
        elif action == "change_cluster":
            player.stop_audio()
            if player.mode == "cluster":
                player.select_initial_cluster()
            else:
                player.init_similar_mode()
        else:
            player.stop_audio()
            continue


if __name__ == "__main__":
    main()
