
## 📟 CLI Usage (`smart_player.py`)

The **CLI Smart Player** is a terminal-based companion to the web player. It uses the same audio feature data, clusters, and similarity engine but runs entirely in the console using Rich-based UI components and optional VLC playback.

This mode is ideal for:

- headless servers (e.g., Raspberry Pi)
- quick debugging of cluster and similarity transitions
- users who prefer keyboard-driven control
- understanding how the similarity engine behaves

---

### 🚀 Starting the CLI Player

```bash
python3 smart_player.py
```

Before running the CLI, make sure the feature CSV exists:

```bash
python3 analyze_and_cluster.py
```

---

### 🎛️ Mode Selection

When the player starts, you can choose between two playback modes:

```
1 → Cluster mode (play songs inside musical groups)
2 → Similarity mode (intelligent transitions based on audio features)
```

#### Cluster Mode
Choose a cluster directly or using filters:

- specific cluster ID
- random cluster
- slow / medium / fast clusters
- filter by:
  - tempo (l/m/s)
  - energy (calm/normal/powerful)
  - brightness (dark/neutral/bright)

This mode focuses on **consistent musical style**.

#### Similarity Mode
Starts with a selected or random song and then automatically transitions to:

- the most similar tracks (smooth flow), or
- very different ones via manual “big jump” requests

Best suited for “smart radio” style playback.

---

### 🎵 During Playback

The CLI displays:

- current track name
- tempo, energy, brightness values
- cluster ID & category labels
- similarity to the previous track (distance + percentage)
- Rich progress bar (if VLC is available)

---

### ⌨️ Controls

| Key | Action |
|------|--------|
| Enter | Skip to next song |
| s | Skip current song |
| g | Skip current cluster / trigger big jump |
| c | Re-select cluster or restart similarity mode |
| q | Quit player |

If VLC is not installed, the script enters **logic-only mode**, still performing transitions without actual playback.

---

### 🔊 VLC Integration

If VLC (`python-vlc`) is installed:

- real playback
- automatic end-of-song detection
- real-time progress feedback

If VLC is missing:

- playback is simulated
- all logic, clustering, and transitions still work

---

### 🧠 Similarity Engine

Similarity mode uses weighted normalized distances across:

- Tempo (weight 2.0)
- RMS Energy (2.0)
- Brightness (1.0)
- MFCC coefficients (0.3 each)

Small distances = very similar  
Large distances = very different

You may trigger “big jumps” at any time.

---

### 📝 Summary

`smart_player.py` provides:

- a full-featured console-based audio explorer
- cluster-driven playback
- feature-based similarity transitions
- VLC-powered playback (optional)
- responsive controls
- perfect complement to the web player
