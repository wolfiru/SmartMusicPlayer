# 🎵 Smart Music Player
### Intelligent Web-Based Music Player with Audio Feature Analysis, Clustering, Mood Presets & Local/Remote Playback

---

## 📌 Overview
The **Smart Music Player** is a powerful web-driven music system designed for desktop and Raspberry Pi environments.  
It analyzes your music library, extracts audio features, clusters your songs, and then uses intelligent similarity algorithms to build continuous playlists based on mood presets or feature proximity.

It supports two playback modes:

- **Local Mode (VLC):** Music plays directly on the host machine  
- **Remote Mode (Browser):** Music streams via `/api/stream/<id>` and plays in the browser’s `<audio>` tag  

The UI is fully responsive and optimized for mobile and desktop environments.

There is also an CLI Version available, see extra file smart_player.

---

## ✨ Key Features

### 🎵 Intelligent Song Selection
- Automatic audio feature extraction using **Librosa**
- Tempo, RMS energy, brightness, MFCC-based timbre analysis
- K-Means clustering (default: 15 clusters)
- Similarity-based "next song" selection
- Optional “Big Jump” for dramatic mood changes

### 🎚️ Mood Presets (presets.json)
Each preset defines:

- Target **Tempo**, **Energy**, **Brightness**
- A weight determining influence on selection
- Stays active until the user manually overrides playback

Example:

```json
{
  "id": 1,
  "name": "Driving",
  "tempo": 65,
  "energy": 70,
  "brightness": 60,
  "weight": 0.9
}
```

### 🎨 Dynamic UI Theme Based on Album Art
- Extracts cover images from:
  - embedded ID3 tags  
  - `cover.jpg`, `folder.png`, etc.
- Computes dominant color per track
- UI buttons & highlights adapt automatically

### 🔊 Two Playback Modes
| Mode | Description |
|------|-------------|
| **Local (VLC)** | Playback happens on server (PC/Raspberry Pi) |
| **Remote (Browser)** | Browser streams MP3 from Flask |

Both modes support **Autoplay**.

---

## 🧠 Architecture

### High-Level Flow
```
Music Directory
      ↓
analyze_and_cluster.py
      ↓ generates
song_features_with_clusters.csv
      ↓
web_player.py (Flask App)
↓            ↓
Local VLC    Remote MP3 Streaming
```

### Tech Stack
- **Python 3**
- **Flask** backend
- **VLC / python-vlc**
- **Mutagen** for album art
- **Pillow (PIL)** for color extraction
- **Librosa** for audio analysis
- **Pandas / NumPy** for data handling
- **HTML/CSS/JavaScript** frontend

---

## 📁 Project Structure

```
SmartMusicPlayer/
│
├── web_player.py                 # Main Flask server + UI logic
├── analyze_and_cluster.py        # Audio analysis & clustering
├── presets.json                  # Mood preset definitions
└── Music/                        # MP3 files
   └── song_features_with_clusters.csv
```

---

# 🔍 Audio Analysis & Clustering (`analyze_and_cluster.py`)

This script analyzes an entire music directory and extracts:

- Tempo (BPM)
- RMS energy level
- Spectral centroid (brightness)
- 13 MFCCs
- K-Means cluster ID

It automatically updates or regenerates:

`MUSIC_DIR/song_features_with_clusters.csv`

### Run:
```bash
python3 analyze_and_cluster.py
```

### What it does:
1. Scans your music folder
2. Extracts audio features with Librosa
3. Computes MFCCs
4. Scales data using StandardScaler
5. Clusters songs into N categories (default 15)
6. Saves final CSV
7. Shows cluster summary in terminal via Rich

All code is included exactly as provided.

---

# 🌐 Web Player (`web_player.py`)

Runs the full player interface:

### Start server:
```bash
python3 web_player.py
```

### Access:
```
http://localhost:5000
http://<raspberry-pi-ip>:5000
```

### UI Features:
- Responsive layout (desktop/mobile)
- Song title above album art
- Cover-based background blur
- Playlist progress bar
- Mood preset selection
- Local/Remote toggle
- Autoplay in both modes
- Similarity metrics displayed

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Player status & metadata |
| `/api/command` | POST | start, next, pause, stop |
| `/api/mode` | GET/POST | switch local/remote modes |
| `/api/presets` | GET | list presets |
| `/api/preset` | POST | activate preset |
| `/api/stream/<id>` | GET | MP3 streaming |
| `/api/cover/<id>` | GET | album cover |
| `/api/remote_ended` | POST | client notifies end of song |

---

# 🍓 Raspberry Pi Support

### Fully supported:
- Raspberry Pi 3 / 4  
- Raspberry Pi OS (Lite or Desktop)
- Headless mode

### Required packages:
```bash
sudo apt update
sudo apt install vlc python3-vlc python3-pil
```

### Audio output selection:
```
sudo raspi-config
→ Advanced Options → Audio
```

---

# 🔧 Installation

### Python dependencies:
```bash
pip install flask mutagen pillow numpy pandas python-vlc librosa scikit-learn rich
```

### Windows VLC Install:
- Install VLC from official site  
- Ensure `libvlc.dll` is in PATH  
- python-vlc will auto-detect it

---

# 📈 Roadmap
- Playlist saving
- Queue editor
- Two-color gradients extracted from album art
- Beat-matched transitions
- Webhooks (HomeAssistant integration)
- Spotify/Youtube metadata importer
- Multi-user mode

---

# 📄 License
MIT License

---

# 🙌 Credits
Project by Wolfgang Ruthner (AT) 
2025 – Present
