# Running SmartMusicPlayer as a systemd Service (Raspberry Pi / Linux)

This guide explains how to run `web_player.py` permanently in the
background using **systemd**. With this setup, the SmartMusicPlayer web
interface starts automatically on boot and restarts if it crashes.

## 1. Requirements

Install required Python modules:

``` bash
sudo apt-get update
sudo apt-get install -y python3-pandas python3-numpy python3-flask python3-pil
sudo apt-get install -y python3-pip
sudo pip3 install mutagen python-vlc rich
```

## 2. File Location

Place the SmartMusicPlayer files in:

    /home/pi/smartmusic/

Ensure `web_player.py` is inside that folder.

## 3. Create the systemd service

``` bash
sudo nano /etc/systemd/system/web_player.service
```

Paste this:

``` ini
[Unit]
Description=SmartMusicPlayer Web Player
After=network.target

[Service]
WorkingDirectory=/home/pi/smartmusic
ExecStart=/usr/bin/python3 /home/pi/smartmusic/web_player.py
User=pi
Group=pi
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=5
Type=simple

[Install]
WantedBy=multi-user.target
```

## 4. Enable and start the service

``` bash
sudo systemctl daemon-reload
sudo systemctl enable web_player.service
sudo systemctl start web_player.service
```

Check status:

``` bash
systemctl status web_player.service
```

## 5. Viewing logs

Live logs:

``` bash
journalctl -u web_player.service -f
```

Last 40 entries:

``` bash
journalctl -u web_player.service -n 40 --no-pager
```

## 6. Stopping or restarting

``` bash
sudo systemctl stop web_player.service
sudo systemctl restart web_player.service
sudo systemctl disable web_player.service
```

## 7. Accessing the Web UI

Default address:

    http://<your-pi-ip>:5000

Your SmartMusicPlayer now runs as a permanent background service.
