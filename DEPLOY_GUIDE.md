# Quick Deploy Guide (You Already Have Streamlit Cloud)

## Step 1: Create GitHub Repo
1. Go to https://github.com/new
2. Name: `swedish-election-sentiment`
3. Make it **PUBLIC**
4. Click **Create repository**

## Step 2: Upload Files
Upload these 3 files from the output folder:
- `app.py`
- `requirements.txt`
- `README.md`

Click "Add file" → "Upload files" → drag & drop → "Commit changes"

## Step 3: Deploy from Streamlit Cloud
1. Go to https://streamlit.io/cloud (you already have an account)
2. Click **"New app"**
3. Select your GitHub repo: `YOUR_USERNAME/swedish-election-sentiment`
4. Branch: `main`
5. Main file path: `app.py`
6. Click **"Deploy"**

## Step 4: Add Secrets
1. In your deployed app, click **"⋮"** → **"Settings"**
2. Go to **"Secrets"**
3. Click **"Edit secrets"**
4. Paste this (replace with your actual keys):

```toml
REDDIT_CLIENT_ID = "your_actual_reddit_client_id"
REDDIT_CLIENT_SECRET = "your_actual_reddit_client_secret"
YOUTUBE_API_KEY = "your_actual_youtube_api_key"
```

5. Click **"Save"**
6. Click **"⋮"** → **"Reboot"**

## Step 5: Done!
Your app is live at: `https://swedish-election-sentiment-xxx.streamlit.app`

---

## Get Your API Keys

### Reddit API (Free)
1. https://www.reddit.com/prefs/apps
2. Click "Create another app..."
3. Type: "script"
4. Name: "Swedish Election Monitor"
5. Redirect URI: `http://localhost:8080`
6. Copy **client_id** (under the app name) and **client_secret**

### YouTube API (Free)
1. https://console.cloud.google.com
2. Create new project → "Swedish Election Monitor"
3. APIs & Services → Library → Search "YouTube Data API v3"
4. Click "Enable"
5. APIs & Services → Credentials → Create Credentials → API Key
6. Copy the key

---

## Using Your Dashboard

1. Open your app URL
2. Click **"Collect New Data"** → waits 2-5 min
3. Click **"Analyze Sentiment"** → waits 3-10 min (downloads BERT model on first run)
4. Review all charts and data
5. Click **"Refresh Dashboard"** anytime to reload

**Note:** First analysis downloads ~500MB multilingual BERT model. Be patient.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Reddit auth failed" | Check secrets spelling. Must be exactly: REDDIT_CLIENT_ID |
| "No data collected" | Your search terms may not match current discussions |
| App shows "waking up" | Normal after 12h inactivity. Wait 30 seconds. |
| "Memory exceeded" | Reduce `reddit_limit` in app.py from 300 to 100 |
| Model download slow | First run only. Wait 5-10 minutes. |
