# 🇸🇪 Swedish Election Sentiment Monitor 2026

A free, AI-powered sentiment monitoring dashboard for the Swedish general election (Riksdagsval) on **September 13, 2026**.

---

## 🚀 Deploy on Streamlit Community Cloud (FREE)

### Step 1: Create Accounts
1. **GitHub:** https://github.com/join (if you don't have one)
2. **Streamlit Community Cloud:** https://streamlit.io/cloud

### Step 2: Create a GitHub Repository
1. Go to https://github.com/new
2. Name it `swedish-election-sentiment`
3. Make it **Public** (required for free tier)
4. Click **Create repository**

### Step 3: Upload Files
Upload these 3 files to your GitHub repo:
- `app.py`
- `requirements.txt`
- `README.md` (this file)

**How:**
- Click "Add file" → "Upload files"
- Drag and drop all 3 files
- Click "Commit changes"

### Step 4: Deploy on Streamlit Community Cloud
1. Go to https://streamlit.io/cloud
2. Click **"New app"**
3. Select your GitHub repo: `YOUR_USERNAME/swedish-election-sentiment`
4. Branch: `main`
5. Main file path: `app.py`
6. Click **"Deploy"**

### Step 5: Add Secrets
1. In your deployed app, click **"⋮"** (top right) → **"Settings"**
2. Go to **"Secrets"** section
3. Add each secret:

| Secret Name | Value | How to Get |
|-------------|-------|------------|
| `REDDIT_CLIENT_ID` | Your Reddit app ID | https://www.reddit.com/prefs/apps |
| `REDDIT_CLIENT_SECRET` | Your Reddit app secret | https://www.reddit.com/prefs/apps |
| `YOUTUBE_API_KEY` | Your YouTube API key | https://console.cloud.google.com |

4. Click **"Save"**
5. Click **"⋮"** → **"Reboot"**

### Step 6: Your Dashboard is Live!
URL: `https://YOUR_APP_NAME.streamlit.app`

---

## 🔄 Using the Dashboard

1. **Collect Data:** Click "Collect New Data" in the sidebar → waits 2-5 min
2. **Analyze Sentiment:** Click "Analyze Sentiment" → waits 3-10 min
3. **View Charts:** Dashboard auto-updates with all visualizations
4. **Refresh:** Click "Refresh Dashboard" to reload data

---

## 📊 What It Monitors

### Swedish Political Parties
| Party | Leader | Bloc |
|-------|--------|------|
| Socialdemokraterna (S) | Magdalena Andersson | Left |
| Moderaterna (M) | Ulf Kristersson | Right |
| Sverigedemokraterna (SD) | Jimmie Åkesson | Right |
| Kristdemokraterna (KD) | Ebba Busch | Right |
| Liberalerna (L) | Johan Pehrson | Right |
| Centerpartiet (C) | Muharrem Demirok | Center |
| Miljöpartiet (MP) | Amanda Lind | Left |
| Vänsterpartiet (V) | Nooshi Dadgostar | Left |

### Key Issues Tracked
- Immigration (invandring)
- Crime (kriminalitet)
- Healthcare (sjukvård)
- Education (skola)
- Economy (ekonomi)
- Climate (klimat)
- NATO
- Housing (bostad)
- Energy prices (elpris)
- Defense (försvar)

---

## 💰 Cost: $0

| Component | Cost |
|-----------|------|
| Streamlit Community Cloud | **FREE** (unlimited public apps) |
| GitHub (public repo) | **FREE** |
| Reddit API | **FREE** |
| YouTube API | **FREE** (10,000 units/day) |
| BERT Model | **FREE** |
| **TOTAL** | **$0** |

---

## ⚠️ Limitations

| Limit | Details |
|-------|---------|
| **Memory** | ~1 GB RAM (enough for BERT + data) |
| **Sleep** | App sleeps after 12 hours of inactivity. Next visitor sees "waking up" page for 30 seconds. |
| **Private apps** | Only 1 private app allowed on free tier. Use public repo. |
| **Custom domain** | Not supported on free tier |
| **Twitter/X** | Not included (free scraping broken) |
| **Facebook** | Limited (requires advanced scraping) |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Reddit auth failed" | Check secrets are set correctly in Streamlit Cloud Settings |
| "No data collected" | Your search terms may not match current discussions. Try broader terms. |
| App shows "waking up" | Normal after 12 hours of inactivity. Wait 30 seconds. |
| "Memory exceeded" | Reduce `reddit_limit` in app.py from 300 to 100 |
| Model download slow | Normal for first build. Wait 5-10 minutes. |

---

## 📅 Election Timeline

- **Election Date:** September 13, 2026
- **Campaign Period:** Typically August - September
- **Early Voting:** Usually starts 18 days before election day

---

## 🆚 Why Streamlit Community Cloud Over Hugging Face?

| Feature | Streamlit Community Cloud | Hugging Face Spaces |
|---------|---------------------------|---------------------|
| **Cost** | ✅ FREE (unlimited public apps) | ⚠️ Gradio/Docker may require PRO |
| **Streamlit native** | ✅ Built by Streamlit team | ❌ Moved to paid Docker |
| **Sleep mode** | After 12 hours | After ~48 hours |
| **Memory** | ~1 GB | ~16 GB (but may be paid) |
| **Setup ease** | Very easy | Moderate |
| **Best for** | Streamlit dashboards | ML model demos |

---

Built with ❤️ for Swedish democracy.
