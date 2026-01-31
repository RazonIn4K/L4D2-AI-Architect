# L4D2 Dedicated Server Setup Guide (Local & Cloud)

This guide covers setting up a Dedicated Server for Left 4 Dead 2, installing SourceMod, and verifying the AI bridge. It supports both **Local Windows/Linux** testing and **DigitalOcean Cloud** deployment.

## 1. Hardware & Cloud Specs

The Source Engine (2009) is **heavily single-threaded**. A CPU with high clock speed (GHz) is far more important than one with many cores.

### DigitalOcean Recommendation

- **Droplet Type:** **Premium CPU-Optimized** (Intel or AMD).
  - _Why?_ Standard/Basic droplets have shared CPUs with lower consistent clock speeds, which will cause lag spikes on a modded server. Premium droplets offer dedicated, high-frequency vCPUs.
- **Size:**
  - **CPU:** 2 vCPUs (1 for game, 1 for OS/system overhead).
  - **RAM:** 4GB (8GB if running many heavy mods).
  - **OS:** Ubuntu 22.04 (LTS) or Debian 12.
- **Estimated Cost:** ~$40-50/mo for a solid modded server (hourly billing available).

---

## 2. Server Installation (Headless Linux / Cloud)

_Run these commands as a non-root user (create one first called `steam` if possible)._

### A. Dependencies

```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y lib32gcc-s1 lib32stdc++6 curl unzip tmux screen
```

### B. Install SteamCMD

```bash
mkdir ~/steamcmd && cd ~/steamcmd
curl -sqL "https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz" | tar zx
```

### C. Install L4D2

```bash
# Install to ~/L4D2Server
./steamcmd.sh +force_install_dir ~/L4D2Server +login anonymous +app_update 222860 validate +quit
```

### D. Initial Test Run

```bash
cd ~/L4D2Server
./srcds_run -game left4dead2 -console -usercon +map c1m1_hotel
# Press Ctrl+C to stop after it loads.
```

---

## 3. Modding Platform (SourceMod)

### A. Download & Install (Scripted)

Run this from your server (`~/L4D2Server`):

```bash
cd ~/L4D2Server/left4dead2

# 1. Metamod: Source (Stable)
curl -L -o mms.tar.gz https://mms.alliedmods.net/mmsdrop/1.11/mmsource-1.11.0-git1148-linux.tar.gz
tar -zxvf mms.tar.gz
rm mms.tar.gz

# 2. SourceMod (Stable)
curl -L -o sm.tar.gz https://sm.alliedmods.net/smdrop/1.12/sourcemod-1.12.0-git6936-linux.tar.gz
tar -zxvf sm.tar.gz
rm sm.tar.gz

# 3. Generate VDF
# (You technically need to generate metamod.vdf, but the default usually works or you can create it:)
echo '"Plugin"
{
	"file"	"../left4dead2/addons/metamod/bin/metamod_2_l4d2"
}' > addons/metamod.vdf
```

---

## 4. Install the AI Bridge Plugin

This is the critical link between your server and the Python AI.

### A. Compile the Plugin

You cannot easily compile `.sp` files on the Linux server without setting up the compiler.

1.  **On your Local Machine**:
    - Go to **[spider.limetech.io](https://spider.limetech.io/)**.
    - Upload/Paste content of: `L4D2-AI-Architect/data/l4d2_server/addons/sourcemod/scripting/l4d2_ai_bridge_v2.sp`
    - Click **Compile**.
    - Download `l4d2_ai_bridge_v2.smx`.

### B. Upload to Server

Use SFTP (FileZilla) or SCP:

```bash
# Upload to your DigitalOcean IP
scp l4d2_ai_bridge_v2.smx root@<Droplet_IP>:~/L4D2Server/left4dead2/addons/sourcemod/plugins/
```

---

## 5. Configuration & Launch

### A. Server Config (`cfg/server.cfg`)

Create/Edit `~/L4D2Server/left4dead2/cfg/server.cfg`:

```cfg
hostname "L4D2 AI Server"
rcon_password "CHANGE_THIS_PASSWORD"
sv_lan 0
sv_cheats 0 (Or 1 if testing)
sv_consistency 0
sv_pure 0
```

### B. Start Script (`start.sh`)

Create `~/L4D2Server/start.sh`:

```bash
#!/bin/bash
# Loop to restart crash
while true
do
    ./srcds_run -game left4dead2 -console -usercon -port 27015 +map c1m1_hotel +host_workshop_collection [YOUR_COLLECTION_ID]
    echo "Server crashed, restarting in 5 seconds..."
    sleep 5
done
```

_Make it executable: `chmod +x start.sh`_

### C. Run in Background

```bash
tmux new -s l4d2
./start.sh
# Press Ctrl+B then D to detach
```

## 6. How to Mod (Remote)

- **Workshop**: Just update your Steam Workshop Collection. The server will download changes on restart/map change.
- **Manual Plugins**: Upload `.smx` files to `addons/sourcemod/plugins/` via SFTP.
