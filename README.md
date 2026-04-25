# Options Signal App — Setup Guide

## What this is
A web app that runs your options signal pipeline (straddle + call+hedge) 
and shows results on a mobile-friendly dashboard. Accessible from any phone.

## Deploy to Railway (free, ~5 minutes)

### Step 1 — Create a GitHub repo

1. Go to https://github.com/new
2. Name it `options-signals` (private is fine)
3. Don't add a README (we'll push files directly)

### Step 2 — Push these files to GitHub

Open Terminal and run:

```bash
cd ~/Desktop
mkdir options-signals-app
cd options-signals-app

# Copy app files into this folder
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/Procfile .
cp /path/to/railway.json .

git init
git add .
git commit -m "Initial deploy"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/options-signals.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 3 — Deploy to Railway

1. Go to https://railway.app and sign up (free, use GitHub login)
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `options-signals` repo
4. Railway auto-detects Python — click **Deploy**
5. Wait ~2 minutes for the build to finish
6. Click **Settings** → **Networking** → **Generate Domain**
7. Your URL will be something like `https://options-signals-production.up.railway.app`

### Step 4 — Bookmark on your phone

Open the URL on your iPhone/Android, tap Share → Add to Home Screen.
It now looks like an app icon.

---

## Usage

- **Run ↗** — tap to run the pipeline manually (takes 1–2 min while it fetches prices)
- **Today tab** — today's signals, green = BUY, gray = no trade
- **Open trades tab** — active positions with hold progress bar
- **History tab** — all closed trades with simulated P&L
- **P&L chart** — cumulative return over time

The pipeline also runs automatically every weekday at 9:05 AM ET.

---

## Railway free tier limits

- 500 hours/month compute (enough for always-on)
- 1 GB memory
- Ephemeral disk — trade/signal data resets on redeploy

To persist data across redeploys, add a Railway Volume:
Settings → Volumes → Add Volume → mount at `/app/data`
Then set env var `DATA_DIR=/app/data`

---

## Optional: Email alerts

In `app.py`, find the CONFIG section and set:
```python
SEND_EMAIL    = True
EMAIL_FROM    = 'you@gmail.com'
EMAIL_TO      = 'you@gmail.com'
EMAIL_PASSWORD = 'xxxx-xxxx-xxxx-xxxx'  # Gmail app password
```

Get a Gmail app password at: https://myaccount.google.com/apppasswords
