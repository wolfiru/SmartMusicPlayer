"""
Music Feature Extraction & Clustering Script

This script scans a directory of audio files, extracts basic audio features,
and groups the songs into clusters using KMeans.

Main features:
- Recursively searches MUSIC_DIR for audio files with extensions:
  .mp3, .wav, .flac, .m4a
- Extracts for each file:
  - tempo (BPM)
  - rms_mean (loudness / energy)
  - spec_centroid_mean (brightness)
  - 13 MFCC mean values (mfcc_1 ... mfcc_13)
- Stores all features in a CSV file (song_features_with_clusters.csv)
- Uses StandardScaler and KMeans to assign each song to a cluster
- Incremental behavior:
  - If the CSV already exists and contains a 'path' column,
    only NEW audio files (not yet in the CSV) are analyzed.
  - Existing feature rows are preserved and re-clustered together with new ones.

Requirements (Python packages):
- librosa
- numpy
- pandas
- scikit-learn
- rich

Configuration:
- Adjust MUSIC_DIR to point to your local music folder.
- N_CLUSTERS controls how many clusters KMeans will create.

Usage:
- Run the script (for example):
    python analyze_and_cluster_songs.py

Output:
- CSV file in MUSIC_DIR:
    song_features_with_clusters.csv
- Columns include:
    path, tempo, rms_mean, spec_centroid_mean,
    mfcc_1 ... mfcc_13, cluster_id
- A cluster summary table is printed to the console.
"""

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

# Make Rich tracebacks prettier
rich_traceback_install()
console = Console()

# === CONFIGURATION ===
MUSIC_DIR = r"M:\Favoriten (Spotify)"   # <-- Adjust path to your folder
OUTPUT_CSV = os.path.join(MUSIC_DIR, "song_features_with_clusters.csv")
N_CLUSTERS = 15  # Number of clusters (can be adjusted later)


def list_audio_files(root_dir, exts=(".mp3", ".wav", ".flac", ".m4a")):
    audio_files = []
    for base, _, names in os.walk(root_dir):
        for n in names:
            if n.lower().endswith(exts):
                audio_files.append(os.path.join(base, n))
    return audio_files


def extract_features(path, sr=22050):
    """
    Extracts basic audio features from a file.
    Returns a dict or None on error.
    """
    try:
        # Load file
        y, sr = librosa.load(path, sr=sr, mono=True)
        if len(y) < sr * 5:
            # Skip very short files
            console.log(f"[yellow]Skipping very short file:[/yellow] {path}")
            return None

        # Tempo / BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # tempo may be array or scalar – ensure a float
        if np.ndim(tempo) > 0:
            tempo_val = float(tempo[0])
        else:
            tempo_val = float(tempo)

        # Loudness / Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms))

        # Brightness
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_centroid_mean = float(np.mean(spec_centroid))

        # MFCCs (timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)  # 13 values

        features = {
            "tempo": tempo_val,
            "rms_mean": rms_mean,
            "spec_centroid_mean": spec_centroid_mean,
        }
        for i, v in enumerate(mfcc_means):
            features[f"mfcc_{i+1}"] = float(v)

        return features

    except Exception as e:
        console.log(f"[red]Error with[/red] {path}: {e}")
        return None


def main():
    console.rule("[bold cyan]Music Analysis & Clustering[/bold cyan]")
    start_time = time.time()

    # 0) Load existing CSV if present
    existing_df = None
    known_paths = set()

    if os.path.exists(OUTPUT_CSV):
        console.print(f"[green]Existing feature file found:[/green] {OUTPUT_CSV}")
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if "path" in existing_df.columns:
                known_paths = set(existing_df["path"].astype(str).tolist())
                console.print(
                    f"Already known songs: [bold]{len(known_paths)}[/bold]"
                )
            else:
                console.print(
                    "[yellow]Warning: existing CSV has no 'path' column. "
                    "It will be ignored and rebuilt.[/yellow]"
                )
                existing_df = None
        except Exception as e:
            console.print(
                f"[red]Could not read existing CSV:[/red] {e}\n"
                "[yellow]Starting with an empty dataset.[/yellow]"
            )
            existing_df = None

    # 1) Collect files
    console.print(f"🎵 Searching audio files in: [bold]{MUSIC_DIR}[/bold]")
    all_audio_files = list_audio_files(MUSIC_DIR)
    total_files = len(all_audio_files)

    if total_files == 0:
        console.print("[red]No audio files found. Check the path![/red]")
        return

    # Only new files not yet in CSV
    new_audio_files = [p for p in all_audio_files if p not in known_paths]
    console.print(f"Total audio files found: [bold]{total_files}[/bold]")
    console.print(f"Files to analyze: [bold green]{len(new_audio_files)}[/bold green]")

    rows_new = []

    # 2) Analysis with progress bar (only new files)
    if new_audio_files:
        console.print("\n[bold cyan]Starting feature analysis for new songs...[/bold cyan]\n")

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "remaining",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing new songs", total=len(new_audio_files))

            for i, path in enumerate(new_audio_files, start=1):
                progress.update(task, description=f"Analyzing: {os.path.basename(path)[:40]}")
                feats = extract_features(path)
                if feats is not None:
                    feats["path"] = path
                    rows_new.append(feats)
                progress.advance(task)

        console.print(
            f"\nNewly analyzed songs: [bold green]{len(rows_new)}[/bold green] "
            f"of [bold]{len(new_audio_files)}[/bold]"
        )
    else:
        console.print("\n[bold yellow]No new songs to analyze.[/bold yellow]")

    # 3) Build DataFrame: existing + new
    if existing_df is not None and not existing_df.empty:
        if rows_new:
            df_new = pd.DataFrame(rows_new)
            df = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            df = existing_df
    else:
        if not rows_new:
            console.print(
                "[red]No existing data and no newly analyzed songs. Aborting.[/red]"
            )
            return
        df = pd.DataFrame(rows_new)

    console.print(
        f"Feature frame size (total): [bold]{df.shape[0]} songs x {df.shape[1]} columns[/bold]"
    )

    # 4) Prepare clustering
    console.print("\n[bold cyan]Scaling features for clustering...[/bold cyan]")
    feature_cols = [c for c in df.columns if c not in ("path", "cluster_id")]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5) KMeans clustering for ALL songs
    console.print(f"[bold cyan]Starting KMeans clustering with {N_CLUSTERS} clusters...[/bold cyan]")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)

    with console.status("[cyan]Computing clusters...[/cyan]", spinner="dots"):
        cluster_ids = kmeans.fit_predict(X_scaled)

    df["cluster_id"] = cluster_ids
    console.print("[green]Clustering completed.[/green]")

    # 6) Save results
    console.print(f"\n[bold cyan]Saving result to:[/bold cyan] {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    console.print("[green]Save completed.[/green]")

    # 7) Pretty cluster overview
    console.print("\n[bold cyan]Cluster overview:[/bold cyan]")
    cluster_counts = df.groupby("cluster_id")["path"].count().reset_index(name="song_count")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Cluster ID", justify="center")
    table.add_column("Song Count", justify="right")

    for _, row in cluster_counts.iterrows():
        cid = str(row["cluster_id"])
        cnt = str(row["song_count"])
        table.add_row(cid, cnt)

    console.print(table)

    # 8) Runtime
    total_time = time.time() - start_time
    console.print(f"\n⏱️ Total runtime: [bold]{total_time:.1f} seconds[/bold]")
    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
