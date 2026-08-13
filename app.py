import os
import sqlite3
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import praw
from googleapiclient.discovery import build
from transformers import pipeline

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

st.set_page_config(page_title="Swedish Election Sentiment 2026", layout="wide")

# ============================================================
# CONFIGURATION
# ============================================================

SWEDISH_PARTIES = {
    "Socialdemokraterna": {"leader": "Magdalena Andersson", "abbrev": "S", "bloc": "left"},
    "Moderaterna": {"leader": "Ulf Kristersson", "abbrev": "M", "bloc": "right"},
    "Sverigedemokraterna": {"leader": "Jimmie Akesson", "abbrev": "SD", "bloc": "right"},
    "Kristdemokraterna": {"leader": "Ebba Busch", "abbrev": "KD", "bloc": "right"},
    "Liberalerna": {"leader": "Johan Pehrson", "abbrev": "L", "bloc": "right"},
    "Centerpartiet": {"leader": "Muharrem Demirok", "abbrev": "C", "bloc": "center"},
    "Miljopartiet": {"leader": "Amanda Lind", "abbrev": "MP", "bloc": "left"},
    "Vansterpartiet": {"leader": "Nooshi Dadgostar", "abbrev": "V", "bloc": "left"},
}

# Try secrets first, then environment variables, then placeholders
try:
    REDDIT_CLIENT_ID = st.secrets.get("REDDIT_CLIENT_ID", os.environ.get("REDDIT_CLIENT_ID", "YOUR_REDDIT_CLIENT_ID"))
    REDDIT_CLIENT_SECRET = st.secrets.get("REDDIT_CLIENT_SECRET", os.environ.get("REDDIT_CLIENT_SECRET", "YOUR_REDDIT_CLIENT_SECRET"))
    YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", os.environ.get("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY"))
except:
    REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "YOUR_REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "YOUR_REDDIT_CLIENT_SECRET")
    YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY")

DB_PATH = "swedish_election_2026.db"

SEARCH_TERMS = [
    "riksdagsval 2026", "val 2026", "Swedish election 2026", "Sverige val",
    "valrorelse 2026", "Socialdemokraterna", "Moderaterna", "Sverigedemokraterna",
    "Kristdemokraterna", "Liberalerna", "Centerpartiet", "Miljopartiet", "Vansterpartiet",
    "Magdalena Andersson", "Ulf Kristersson", "Jimmie Akesson", "Ebba Busch",
    "invandring", "kriminalitet", "sjukvard", "skola", "ekonomi", "klimat", "NATO",
    "bidrag", "bostad", "elpris", "forsvar",
]

SUBREDDITS = ["sweden", "svenskpolitik", "svenska", "europe", "worldnews", "politics"]
YOUTUBE_QUERIES = [
    "riksdagsval 2026", "svensk politik 2026", "valdebatt 2026",
    "Magdalena Andersson", "Ulf Kristersson", "Jimmie Akesson", "Sverigedemokraterna"
]

# ============================================================
# DATABASE SETUP
# ============================================================

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reddit_posts (
            id TEXT PRIMARY KEY, source TEXT DEFAULT 'reddit', subreddit TEXT,
            author TEXT, title TEXT, text TEXT, score INTEGER, num_comments INTEGER,
            created_utc REAL, url TEXT, permalink TEXT,
            sentiment_label TEXT, sentiment_score REAL, party_mentioned TEXT, issue_mentioned TEXT,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS youtube_comments (
            id TEXT PRIMARY KEY, source TEXT DEFAULT 'youtube', video_id TEXT,
            video_title TEXT, author TEXT, text TEXT, like_count INTEGER,
            published_at TEXT, sentiment_label TEXT, sentiment_score REAL,
            party_mentioned TEXT, issue_mentioned TEXT,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_summary (
            date TEXT PRIMARY KEY, total_posts INTEGER, positive_count INTEGER,
            negative_count INTEGER, neutral_count INTEGER, avg_sentiment REAL,
            top_positive TEXT, top_negative TEXT
        )
    """)
    conn.commit()
    conn.close()

# ============================================================
# HELPERS
# ============================================================

def detect_party(text):
    if not text: return None
    text_lower = text.lower()
    party_keywords = {
        "Socialdemokraterna": ["socialdemokraterna", "socialdemokrat", "sosse", "magdalena andersson", "s-partiet"],
        "Moderaterna": ["moderaterna", "moderat", "ulf kristersson", "m-partiet"],
        "Sverigedemokraterna": ["sverigedemokraterna", "sverigedemokrat", "sd", "jimmie akesson", "jimmie åkesson"],
        "Kristdemokraterna": ["kristdemokraterna", "kristdemokrat", "kd", "ebba busch"],
        "Liberalerna": ["liberalerna", "liberal", "folkpartiet", "johan pehrson", "fp"],
        "Centerpartiet": ["centerpartiet", "centerparti", "center", "c-partiet", "muharrem demirok"],
        "Miljopartiet": ["miljopartiet", "miljöpartiet", "miljoparti", "miljöparti", "mp", "amanda lind"],
        "Vansterpartiet": ["vansterpartiet", "vänsterpartiet", "vansterparti", "vänsterparti", "vänster", "nooshi dadgostar", "v-partiet"],
    }
    for party, keywords in party_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return party
    return None

def detect_issue(text):
    if not text: return None
    text_lower = text.lower()
    issue_keywords = {
        "Immigration": ["invandring", "immigration", "migrant", "flykting", "asyl", "integration"],
        "Crime": ["kriminalitet", "brott", "crime", "vald", "våld", "skjutning", "gang", "gäng"],
        "Healthcare": ["sjukvard", "sjukvård", "vard", "vård", "healthcare", "sjukhus", "lakare", "läkare"],
        "Education": ["skola", "utbildning", "school", "larare", "lärare", "elever"],
        "Economy": ["ekonomi", "economy", "inflation", "priser", "bnp", "recession"],
        "Climate": ["klimat", "climate", "miljo", "miljö", "koldioxid"],
        "NATO": ["nato", "forsvar", "försvar", "defense", "militar", "militär"],
        "Housing": ["bostad", "housing", "bostader", "bostäder", "hyra", "bostadsbrist"],
        "Energy": ["elpris", "energi", "energy", "el", "karnkraft", "kärnkraft", "vindkraft"],
        "Welfare": ["bidrag", "welfare", "forsorjningsstod", "försörjningsstöd", "socialbidrag", "pension"],
    }
    for issue, keywords in issue_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return issue
    return None

# ============================================================
# AI EMPLOYEE #1: SCOUT
# ============================================================

def collect_reddit(limit=300):
    if REDDIT_CLIENT_ID == "YOUR_REDDIT_CLIENT_ID":
        return 0, "ERROR: Reddit API credentials not configured. Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET as Secrets."
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent="SwedishElectionMonitor/1.0")
    except Exception as e:
        return 0, f"ERROR: Reddit auth failed: {e}"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    count = 0
    for subreddit_name in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(subreddit_name)
        except:
            continue
        for term in SEARCH_TERMS:
            try:
                limit_per_term = max(1, limit // len(SEARCH_TERMS))
                for post in subreddit.search(term, limit=limit_per_term):
                    party = detect_party(post.title + " " + post.selftext)
                    issue = detect_issue(post.title + " " + post.selftext)
                    cursor.execute("""
                        INSERT OR IGNORE INTO reddit_posts 
                        (id, subreddit, author, title, text, score, num_comments, created_utc, url, permalink, party_mentioned, issue_mentioned)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (post.id, subreddit_name, str(post.author), post.title, post.selftext,
                          post.score, post.num_comments, post.created_utc, post.url, post.permalink, party, issue))
                    count += 1
            except:
                continue
    conn.commit()
    conn.close()
    return count, f"Collected {count} Reddit posts"

def collect_youtube(max_results=30):
    if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY":
        return 0, "ERROR: YouTube API key not configured. Add YOUTUBE_API_KEY as Secret."
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        return 0, f"ERROR: YouTube API init failed: {e}"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    count = 0
    for query in YOUTUBE_QUERIES:
        try:
            search_response = youtube.search().list(q=query, part='id,snippet', maxResults=max_results, type='video', order='relevance').execute()
            for video in search_response.get('items', []):
                video_id = video['id']['videoId']
                video_title = video['snippet']['title']
                try:
                    comments_response = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=50, order='relevance').execute()
                    for item in comments_response.get('items', []):
                        comment = item['snippet']['topLevelComment']['snippet']
                        comment_id = item['id']
                        party = detect_party(comment['textDisplay'])
                        issue = detect_issue(comment['textDisplay'])
                        cursor.execute("""
                            INSERT OR IGNORE INTO youtube_comments
                            (id, video_id, video_title, author, text, like_count, published_at, party_mentioned, issue_mentioned)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (comment_id, video_id, video_title, comment['authorDisplayName'],
                              comment['textDisplay'], comment['likeCount'], comment['publishedAt'], party, issue))
                        count += 1
                except:
                    continue
        except:
            continue
    conn.commit()
    conn.close()
    return count, f"Collected {count} YouTube comments"

# ============================================================
# AI EMPLOYEE #2: JUDGE
# ============================================================

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def analyze_database():
    classifier = load_sentiment_model()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, text FROM reddit_posts WHERE sentiment_label IS NULL")
    reddit_posts = cursor.fetchall()
    for post_id, title, text in reddit_posts:
        full_text = f"{title} {text}" if text else title
        if not full_text or len(full_text.strip()) == 0:
            label, score = "neutral", 0.0
        else:
            try:
                result = classifier(full_text[:512])[0]
                label = result['label'].lower()
                score = result['score']
                if 'positive' in label: label, score = 'positive', score
                elif 'negative' in label: label, score = 'negative', -score
                else: label, score = 'neutral', 0.0
            except:
                label, score = "neutral", 0.0
        cursor.execute("UPDATE reddit_posts SET sentiment_label=?, sentiment_score=? WHERE id=?", (label, score, post_id))
    cursor.execute("SELECT id, text FROM youtube_comments WHERE sentiment_label IS NULL")
    youtube_comments = cursor.fetchall()
    for comment_id, text in youtube_comments:
        if not text or len(text.strip()) == 0:
            label, score = "neutral", 0.0
        else:
            try:
                result = classifier(text[:512])[0]
                label = result['label'].lower()
                score = result['score']
                if 'positive' in label: label, score = 'positive', score
                elif 'negative' in label: label, score = 'negative', -score
                else: label, score = 'neutral', 0.0
            except:
                label, score = "neutral", 0.0
        cursor.execute("UPDATE youtube_comments SET sentiment_label=?, sentiment_score=? WHERE id=?", (label, score, comment_id))
    conn.commit()
    conn.close()
    return len(reddit_posts) + len(youtube_comments)

def generate_summary():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT sentiment_label, sentiment_score, text, party_mentioned, issue_mentioned, 'reddit' as source 
        FROM reddit_posts WHERE sentiment_label IS NOT NULL
        UNION ALL
        SELECT sentiment_label, sentiment_score, text, party_mentioned, issue_mentioned, 'youtube' as source 
        FROM youtube_comments WHERE sentiment_label IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    if len(df) == 0:
        return None
    summary = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_posts': len(df),
        'positive_count': len(df[df['sentiment_label'] == 'positive']),
        'negative_count': len(df[df['sentiment_label'] == 'negative']),
        'neutral_count': len(df[df['sentiment_label'] == 'neutral']),
        'avg_sentiment': df['sentiment_score'].mean(),
    }
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO sentiment_summary 
        (date, total_posts, positive_count, negative_count, neutral_count, avg_sentiment, top_positive, top_negative)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (summary['date'], summary['total_posts'], summary['positive_count'], summary['negative_count'],
          summary['neutral_count'], summary['avg_sentiment'], "N/A", "N/A"))
    conn.commit()
    conn.close()
    return summary

# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard():
    st.title("🇸🇪 Swedish Election Sentiment Monitor 2026")
    days_until = (datetime(2026, 9, 13) - datetime.now()).days
    st.markdown(f"**Riksdagsval: September 13, 2026** | {days_until} days until election")
    st.markdown("AI-powered social media sentiment analysis for the Swedish general election")

    with st.sidebar:
        st.title("🎛️ Controls")
        if st.button("🔄 Collect New Data", use_container_width=True):
            with st.spinner("Collecting from Reddit and YouTube..."):
                reddit_count, reddit_msg = collect_reddit()
                youtube_count, youtube_msg = collect_youtube()
                st.success(f"{reddit_msg} | {youtube_msg}")
        if st.button("🧠 Analyze Sentiment", use_container_width=True):
            with st.spinner("Running BERT sentiment analysis..."):
                analyzed = analyze_database()
                summary = generate_summary()
                st.success(f"Analyzed {analyzed} items")
        if st.button("📊 Refresh Dashboard", use_container_width=True):
            st.rerun()
        st.markdown("---")
        st.subheader("Party Reference")
        for party, info in SWEDISH_PARTIES.items():
            st.markdown(f"**{info['abbrev']}** - {party}")

    conn = sqlite3.connect(DB_PATH)
    reddit_df = pd.read_sql_query("SELECT * FROM reddit_posts WHERE sentiment_label IS NOT NULL", conn)
    youtube_df = pd.read_sql_query("SELECT * FROM youtube_comments WHERE sentiment_label IS NOT NULL", conn)
    summary_df = pd.read_sql_query("SELECT * FROM sentiment_summary ORDER BY date DESC LIMIT 30", conn)
    conn.close()

    reddit_df['source'] = 'Reddit'
    youtube_df['source'] = 'YouTube'
    if len(reddit_df) > 0:
        reddit_df['text'] = reddit_df['title'].fillna('') + ' ' + reddit_df['text'].fillna('')

    all_data = pd.concat([
        reddit_df[['source', 'text', 'sentiment_label', 'sentiment_score', 'collected_at', 'party_mentioned', 'issue_mentioned']],
        youtube_df[['source', 'text', 'sentiment_label', 'sentiment_score', 'collected_at', 'party_mentioned', 'issue_mentioned']]
    ], ignore_index=True)

    if len(all_data) == 0:
        st.info("No data yet. Click 'Collect New Data' and 'Analyze Sentiment' in the sidebar to get started.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Posts", len(all_data))
    with col2:
        pos_pct = len(all_data[all_data['sentiment_label'] == 'positive']) / len(all_data) * 100
        st.metric("Positive %", f"{pos_pct:.1f}%")
    with col3:
        neg_pct = len(all_data[all_data['sentiment_label'] == 'negative']) / len(all_data) * 100
        st.metric("Negative %", f"{neg_pct:.1f}%")
    with col4:
        avg_score = all_data['sentiment_score'].mean()
        st.metric("Avg Sentiment", f"{avg_score:.3f}")
    with col5:
        total_parties = all_data['party_mentioned'].notna().sum()
        st.metric("Party Mentions", total_parties)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Sentiment Distribution")
        sentiment_counts = all_data['sentiment_label'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                    color=sentiment_counts.index,
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.subheader("Sentiment by Source")
        source_sentiment = all_data.groupby(['source', 'sentiment_label']).size().reset_index(name='count')
        fig = px.bar(source_sentiment, x='source', y='count', color='sentiment_label',
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
        st.plotly_chart(fig, use_container_width=True)

    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.subheader("Sentiment by Party")
        party_data = all_data[all_data['party_mentioned'].notna()]
        if len(party_data) > 0:
            party_sentiment = party_data.groupby(['party_mentioned', 'sentiment_label']).size().reset_index(name='count')
            fig = px.bar(party_sentiment, x='party_mentioned', y='count', color='sentiment_label',
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
            fig.update_layout(xaxis_title="Party", yaxis_title="Mentions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No party mentions found yet.")
    with col_right2:
        st.subheader("Top Issues Discussed")
        issue_data = all_data[all_data['issue_mentioned'].notna()]
        if len(issue_data) > 0:
            issue_counts = issue_data['issue_mentioned'].value_counts().head(10)
            fig = px.bar(x=issue_counts.index, y=issue_counts.values, labels={'x': 'Issue', 'y': 'Mentions'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No issue mentions found yet.")

    st.subheader("Sentiment Trend Over Time")
    if len(summary_df) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=summary_df['date'], y=summary_df['avg_sentiment'],
                                  mode='lines+markers', name='Avg Sentiment',
                                  line=dict(color='#3498db', width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_title="Date", yaxis_title="Sentiment Score")
        st.plotly_chart(fig, use_container_width=True)

    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        st.subheader("Word Cloud - Negative Posts")
        negative_text = ' '.join(all_data[all_data['sentiment_label'] == 'negative']['text'].dropna().astype(str))
        if negative_text:
            wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(negative_text)
            st.image(wordcloud.to_array(), use_container_width=True)
    with col_wc2:
        st.subheader("Word Cloud - Positive Posts")
        positive_text = ' '.join(all_data[all_data['sentiment_label'] == 'positive']['text'].dropna().astype(str))
        if positive_text:
            wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(positive_text)
            st.image(wordcloud.to_array(), use_container_width=True)

    st.subheader("Recent Posts")
    display_df = all_data[['collected_at', 'source', 'sentiment_label', 'sentiment_score', 'party_mentioned', 'issue_mentioned', 'text']].tail(50)
    display_df = display_df.sort_values('collected_at', ascending=False)
    st.dataframe(display_df, use_container_width=True)

    st.subheader("Download Data")
    csv = all_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "swedish_election_sentiment.csv", "text/csv")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    init_database()
    show_dashboard()
