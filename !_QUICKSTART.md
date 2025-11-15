## 🚀 Quickstart – Installation & erster Start

### 1. Repository holen

```bash
git clone https://github.com/wolfiru/SmartMusicPlayer.git
cd SmartMusicPlayer
```

### 2. Python-Umgebung vorbereiten (optional, empfohlen)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 3. Abhängigkeiten installieren

```bash
pip install --upgrade pip
pip install flask mutagen pillow numpy pandas python-vlc librosa scikit-learn rich
```

> 💡 Unter Linux / Raspberry Pi ggf. zusätzlich:
> ```bash
> sudo apt update
> sudo apt install vlc python3-vlc python3-pil
> ```

### 4. Musik-Verzeichnis anpassen

In folgenden Dateien den Pfad zu deinem Musikordner setzen:

- `analyze_and_cluster.py`
- `web_player.py`
- `smart_player.py`

### 5. Songs analysieren & Cluster erzeugen

```bash
python analyze_and_cluster.py
```

### 6. Web-Player starten

```bash
python web_player.py
```

### 7. CLI-Player starten

```bash
python smart_player.py
```
