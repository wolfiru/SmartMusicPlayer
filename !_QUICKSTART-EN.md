# 🚀 Quickstart – Installation & First Steps

### 1. Clone the repository

```bash
git clone https://github.com/wolfiru/SmartMusicPlayer.git
cd SmartMusicPlayer
```

### 2. Create a Python virtual environment (optional but recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install required dependencies

```bash
pip install --upgrade pip
pip install flask mutagen pillow numpy pandas python-vlc librosa scikit-learn rich
```

> 💡 On Linux / Raspberry Pi you may also need:
> ```bash
> sudo apt update
> sudo apt install vlc python3-vlc python3-pil
> ```

### 4. Configure your music directory

Open these files and adjust the path to your music folder:

- `analyze_and_cluster.py`
- `web_player.py`
- `smart_player.py`

Example:

```python
MUSIC_DIR = r"M:\Favoriten (Spotify)"   # Windows
# or:
MUSIC_DIR = r"/home/pi/music"           # Linux / Raspberry Pi
```

### 5. Analyze songs & generate clusters

```bash
python analyze_and_cluster.py
```

This script will:

- scan your music directory  
- extract audio features (tempo, energy, brightness, MFCCs)  
- perform clustering  
- write/update `song_features_with_clusters.csv`

### 6. Start the Web Player

```bash
python web_player.py
```

Then open:

- On the same machine → `http://localhost:5000`
- From another device on your network → `http://<server-ip>:5000`

The Web Player lets you:

- choose presets (moods / scenarios)  
- switch between *local VLC playback* and *remote browser playback*  
- view covers, progress, similarity, stats, etc.

### 7. Start the CLI Player

```bash
python smart_player.py
```

Features:

- **Cluster Mode** – plays songs within a cluster (mood group)  
- **Similarity Mode** – intelligently transitions between songs using feature distances  
- Keyboard controls:
  - `Enter` → next song  
  - `s` → skip  
  - `g` → big jump / change mood  
  - `c` → reselect cluster / restart mode  
  - `q` → quit  

---

You're ready to go — enjoy your intelligent music player! 🎵
