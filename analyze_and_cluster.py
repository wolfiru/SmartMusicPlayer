import os
import time

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.traceback import install as rich_traceback_install

# Rich-Tracebacks schöner machen
rich_traceback_install()
console = Console()

# === KONFIGURATION ===
MUSIC_DIR = r"M:\Favoriten (Spotify)"   # <-- Pfad zu deinem Ordner anpassen
OUTPUT_CSV = os.path.join(MUSIC_DIR, "song_features_with_clusters.csv")
N_CLUSTERS = 15  # Anzahl der Gruppen (später anpassbar)


def list_audio_files(root_dir, exts=(".mp3", ".wav", ".flac", ".m4a")):
    audio_files = []
    for base, _, names in os.walk(root_dir):
        for n in names:
            if n.lower().endswith(exts):
                audio_files.append(os.path.join(base, n))
    return audio_files


def extract_features(path, sr=22050):
    """
    Extrahiert grundlegende Audio-Features aus einer Datei.
    Gibt ein dict zurück oder None bei Fehler.
    """
    try:
        # Datei laden
        y, sr = librosa.load(path, sr=sr, mono=True)
        if len(y) < sr * 5:
            # sehr kurze Files überspringen
            console.log(f"[yellow]Überspringe sehr kurzes File:[/yellow] {path}")
            return None

        # Tempo/BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # tempo kann Array oder Skalar sein – wir holen uns explizit einen Float
        if np.ndim(tempo) > 0:
            tempo_val = float(tempo[0])
        else:
            tempo_val = float(tempo)

        # Lautheit / Energie
        rms = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms))

        # Helligkeit
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_centroid_mean = float(np.mean(spec_centroid))

        # MFCCs (Klangfarbe)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)  # 13 Werte

        features = {
            "tempo": tempo_val,
            "rms_mean": rms_mean,
            "spec_centroid_mean": spec_centroid_mean,
        }
        for i, v in enumerate(mfcc_means):
            features[f"mfcc_{i+1}"] = float(v)

        return features

    except Exception as e:
        console.log(f"[red]Fehler bei[/red] {path}: {e}")
        return None


def main():
    console.rule("[bold cyan]Musikanalyse & Clustering[/bold cyan]")
    start_time = time.time()

    # 0) Evtl. bestehende CSV laden
    existing_df = None
    known_paths = set()

    if os.path.exists(OUTPUT_CSV):
        console.print(f"[green]Bestehende Feature-Datei gefunden:[/green] {OUTPUT_CSV}")
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if "path" in existing_df.columns:
                known_paths = set(existing_df["path"].astype(str).tolist())
                console.print(
                    f"Bereits bekannte Songs: [bold]{len(known_paths)}[/bold]"
                )
            else:
                console.print(
                    "[yellow]Warnung: bestehende CSV hat keine 'path'-Spalte. "
                    "Sie wird ignoriert und neu aufgebaut.[/yellow]"
                )
                existing_df = None
        except Exception as e:
            console.print(
                f"[red]Konnte bestehende CSV nicht lesen:[/red] {e}\n"
                "[yellow]Starte mit leerem Datensatz.[/yellow]"
            )
            existing_df = None

    # 1) Dateien einsammeln
    console.print(f"🎵 Suche Audiodateien in: [bold]{MUSIC_DIR}[/bold]")
    all_audio_files = list_audio_files(MUSIC_DIR)
    total_files = len(all_audio_files)

    if total_files == 0:
        console.print("[red]Keine Audiodateien gefunden. Pfad prüfen![/red]")
        return

    # Nur neue Dateien (noch nicht in CSV) analysieren
    new_audio_files = [p for p in all_audio_files if p not in known_paths]
    console.print(f"Gesamt gefundene Audiodateien: [bold]{total_files}[/bold]")
    console.print(f"Hiervon neu zu analysieren: [bold green]{len(new_audio_files)}[/bold green]")

    rows_new = []

    # 2) Analyse mit Fortschrittsbalken (nur neue Dateien)
    if new_audio_files:
        console.print("\n[bold cyan]Starte Feature-Analyse für neue Songs...[/bold cyan]\n")
        from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

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
            task = progress.add_task("Analysiere neue Songs", total=len(new_audio_files))

            for i, path in enumerate(new_audio_files, start=1):
                progress.update(task, description=f"Analysiere: {os.path.basename(path)[:40]}")
                feats = extract_features(path)
                if feats is not None:
                    feats["path"] = path
                    rows_new.append(feats)
                progress.advance(task)

        console.print(
            f"\nNeu analysierte Songs: [bold green]{len(rows_new)}[/bold green] "
            f"von [bold]{len(new_audio_files)}[/bold]"
        )
    else:
        console.print("\n[bold yellow]Keine neuen Songs zu analysieren.[/bold yellow]")

    # 3) DataFrame zusammenbauen: bestehend + neu
    if existing_df is not None and not existing_df.empty:
        if rows_new:
            df_new = pd.DataFrame(rows_new)
            # Spalten angleichen (falls sich etwas geändert hat)
            df = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            df = existing_df
    else:
        if not rows_new:
            console.print(
                "[red]Es gibt weder bestehende Daten noch neue analysierte Songs. Abbruch.[/red]"
            )
            return
        df = pd.DataFrame(rows_new)

    console.print(
        f"Feature-Frame Größe (gesamt): [bold]{df.shape[0]} Songs x {df.shape[1]} Spalten[/bold]"
    )

    # 4) Clustering vorbereiten
    console.print("\n[bold cyan]Skaliere Features für Clustering...[/bold cyan]")
    feature_cols = [c for c in df.columns if c not in ("path", "cluster_id")]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5) Clustering mit k-means über ALLE Songs
    console.print(f"[bold cyan]Starte KMeans-Clustering mit {N_CLUSTERS} Clustern...[/bold cyan]")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)

    with console.status("[cyan]Berechne Cluster...[/cyan]", spinner="dots"):
        cluster_ids = kmeans.fit_predict(X_scaled)

    df["cluster_id"] = cluster_ids
    console.print("[green]Clustering abgeschlossen.[/green]")

    # 6) Ergebnisse speichern (im Musikordner)
    console.print(f"\n[bold cyan]Speichere Ergebnis in:[/bold cyan] {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    console.print("[green]Speichern abgeschlossen.[/green]")

    # 7) Cluster-Übersicht hübsch anzeigen
    console.print("\n[bold cyan]Cluster-Übersicht:[/bold cyan]")
    cluster_counts = df.groupby("cluster_id")["path"].count().reset_index(name="anzahl_songs")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Cluster-ID", justify="center")
    table.add_column("Anzahl Songs", justify="right")

    for _, row in cluster_counts.iterrows():
        cid = str(row["cluster_id"])
        cnt = str(row["anzahl_songs"])
        table.add_row(cid, cnt)

    console.print(table)

    # 8) Laufzeit anzeigen
    total_time = time.time() - start_time
    console.print(f"\n⏱️ Gesamtlaufzeit: [bold]{total_time:.1f} Sekunden[/bold]")
    console.rule("[bold green]Fertig[/bold green]")


if __name__ == "__main__":
    main()
