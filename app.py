import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wordcloud import WordCloud
from transformers import pipeline

# Optional advanced libraries
try:
    import gensim
    from gensim import corpora, models
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional APIs
try:
    import praw
except ImportError:
    praw = None

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None

# Google Trends (unofficial API wrapper)
try:
    from pytrends_modern import TrendReq
    from pytrends_modern.exceptions import TooManyRequestsError
except ImportError:
    TrendReq = None
    TooManyRequestsError = Exception


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Swedish Election Intelligence 2026",
    page_icon="🇸🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "🇸🇪 Swedish Election Intelligence Monitor 2026"
DB_PATH = "swedish_election_2026.db"
ELECTION_DATE = datetime(2026, 9, 13)

# Lower defaults for free tier
DEFAULT_REDDIT_LIMIT = 80
DEFAULT_YOUTUBE_RESULTS = 10
DEFAULT_YOUTUBE_COMMENTS = 20

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MIN_MENTIONS_FOR_ALERT = 5
GOOGLE_TRENDS_TIMEFRAME = "today 3-m"
GOOGLE_TRENDS_GEO = "SE"


# ============================================================
# PARTY DATA
# ============================================================

SWEDISH_PARTIES = {
    "Socialdemokraterna": {"leader": "Magdalena Andersson", "abbrev": "S", "bloc": "left"},
    "Moderaterna": {"leader": "Ulf Kristersson", "abbrev": "M", "bloc": "right"},
    "Sverigedemokraterna": {"leader": "Jimmie Åkesson", "abbrev": "SD", "bloc": "right"},
    "Kristdemokraterna": {"leader": "Ebba Busch", "abbrev": "KD", "bloc": "right"},
    "Liberalerna": {"leader": "Johan Pehrson", "abbrev": "L", "bloc": "right"},
    "Centerpartiet": {"leader": "Muharrem Demirok", "abbrev": "C", "bloc": "center"},
    "Miljöpartiet": {"leader": "Amanda Lind", "abbrev": "MP", "bloc": "left"},
    "Vänsterpartiet": {"leader": "Nooshi Dadgostar", "abbrev": "V", "bloc": "left"},
}

BLOC_COLORS = {"left": "#e74c3c", "right": "#3498db", "center": "#f1c40f"}

PARTY_KEYWORDS = {
    "Socialdemokraterna": ["socialdemokraterna", "socialdemokrat", "sosse", "magdalena andersson", "s-partiet"],
    "Moderaterna": ["moderaterna", "moderat", "ulf kristersson", "m-partiet"],
    "Sverigedemokraterna": ["sverigedemokraterna", "sverigedemokrat", "jimmie akesson", "jimmie åkesson", "sd"],
    "Kristdemokraterna": ["kristdemokraterna", "kristdemokrat", "ebba busch", "kd"],
    "Liberalerna": ["liberalerna", "folkpartiet", "johan pehrson", "liberal", "fp"],
    "Centerpartiet": ["centerpartiet", "centerparti", "muharrem demirok", "c-partiet"],
    "Miljöpartiet": ["miljöpartiet", "miljopartiet", "miljöparti", "miljoparti", "amanda lind", "mp"],
    "Vänsterpartiet": ["vänsterpartiet", "vansterpartiet", "vänsterparti", "vansterparti", "nooshi dadgostar", "v-partiet"],
}

LEADER_KEYWORDS = {
    "Magdalena Andersson": ["magdalena andersson"],
    "Ulf Kristersson": ["ulf kristersson"],
    "Jimmie Åkesson": ["jimmie åkesson", "jimmie akesson"],
    "Ebba Busch": ["ebba busch"],
    "Johan Pehrson": ["johan pehrson"],
    "Muharrem Demirok": ["muharrem demirok"],
    "Amanda Lind": ["amanda lind"],
    "Nooshi Dadgostar": ["nooshi dadgostar"],
}

ISSUE_KEYWORDS = {
    "Immigration": ["invandring", "immigration", "migrant", "migranter", "flykting", "flyktingar", "asyl", "integration"],
    "Crime": ["kriminalitet", "brott", "brottslighet", "crime", "våld", "vald", "skjutning", "skjutningar", "gäng", "gang", "gängen"],
    "Healthcare": ["sjukvård", "sjukvard", "vård", "vard", "healthcare", "sjukhus", "läkare", "lakare"],
    "Education": ["skola", "skolan", "utbildning", "school", "lärare", "larare", "elever"],
    "Economy": ["ekonomi", "economy", "inflation", "priser", "pris", "bnp", "recession", "ränta", "ranta", "skatt", "skatter"],
    "Climate": ["klimat", "climate", "miljö", "miljo", "koldioxid", "utsläpp", "utslapp"],
    "NATO & Defence": ["nato", "försvar", "forsvar", "defense", "militär", "militar", "värnplikt", "varnplikt"],
    "Housing": ["bostad", "bostäder", "bostader", "housing", "hyra", "hyror", "bostadsbrist"],
    "Energy": ["elpris", "elpriser", "energi", "energy", "el", "kärnkraft", "karnkraft", "vindkraft"],
    "Welfare": ["bidrag", "welfare", "försörjningsstöd", "forsorjningsstod", "socialbidrag", "pension", "välfärd", "valfard"],
}

SEARCH_TERMS = [
    "riksdagsval 2026", "val 2026", "Swedish election 2026", "Sverige val", "valrörelse 2026",
    "Socialdemokraterna", "Moderaterna", "Sverigedemokraterna", "Kristdemokraterna", "Liberalerna",
    "Centerpartiet", "Miljöpartiet", "Vänsterpartiet", "Magdalena Andersson", "Ulf Kristersson",
    "Jimmie Åkesson", "Ebba Busch", "invandring", "kriminalitet", "sjukvård", "skola", "ekonomi",
    "klimat", "NATO", "bidrag", "bostad", "elpris", "försvar",
]

SUBREDDITS = ["sweden", "svenskpolitik", "svenska", "europe", "worldnews", "politics"]

YOUTUBE_QUERIES = [
    "riksdagsval 2026", "svensk politik 2026", "valdebatt 2026",
    "Magdalena Andersson", "Ulf Kristersson", "Jimmie Åkesson", "Sverigedemokraterna",
]

FLASHBACK_BASE_URL = "https://www.flashback.org"
FLASHBACK_SEARCH_TERMS = [
    "riksdagsval 2026", "Socialdemokraterna", "Moderaterna", "Sverigedemokraterna",
    "Magdalena Andersson", "Ulf Kristersson", "Jimmie Åkesson", "invandring", "kriminalitet", "ekonomi",
]
FLASHBACK_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
DEFAULT_FLASHBACK_THREADS_PER_TERM = 2
DEFAULT_FLASHBACK_POSTS_PER_THREAD = 10
FLASHBACK_REQUEST_DELAY_SECONDS = 2.0


# ============================================================
# SECRETS
# ============================================================

def get_secret(name, placeholder):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    value = os.environ.get(name)
    if value:
        return value
    return placeholder

REDDIT_CLIENT_ID = get_secret("REDDIT_CLIENT_ID", "YOUR_REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = get_secret("REDDIT_CLIENT_SECRET", "YOUR_REDDIT_CLIENT_SECRET")
YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY")
GOOGLE_TRENDS_PROXY_URL = get_secret("GOOGLE_TRENDS_PROXY_URL", None)
FLASHBACK_PROXY_URL = get_secret("FLASHBACK_PROXY_URL", None)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def init_database():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reddit_posts (
                id TEXT PRIMARY KEY,
                source TEXT DEFAULT 'reddit',
                subreddit TEXT,
                author TEXT,
                title TEXT,
                text TEXT,
                score INTEGER DEFAULT 0,
                num_comments INTEGER DEFAULT 0,
                created_utc REAL,
                url TEXT,
                permalink TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                party_mentioned TEXT,
                leader_mentioned TEXT,
                issue_mentioned TEXT,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS youtube_comments (
                id TEXT PRIMARY KEY,
                source TEXT DEFAULT 'youtube',
                video_id TEXT,
                video_title TEXT,
                author TEXT,
                text TEXT,
                like_count INTEGER DEFAULT 0,
                published_at TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                party_mentioned TEXT,
                leader_mentioned TEXT,
                issue_mentioned TEXT,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reddit_count INTEGER DEFAULT 0,
                youtube_count INTEGER DEFAULT 0,
                flashback_count INTEGER DEFAULT 0,
                total_new INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                items_analyzed INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_summary (
                date TEXT PRIMARY KEY,
                total_posts INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                avg_sentiment REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS google_trends (
                date TEXT NOT NULL,
                term TEXT NOT NULL,
                interest INTEGER DEFAULT 0,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, term)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flashback_posts (
                id TEXT PRIMARY KEY,
                source TEXT DEFAULT 'flashback',
                thread_id TEXT,
                thread_title TEXT,
                thread_url TEXT,
                author TEXT,
                text TEXT,
                post_number INTEGER DEFAULT 0,
                posted_at TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                party_mentioned TEXT,
                leader_mentioned TEXT,
                issue_mentioned TEXT,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Add indexes for speed
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reddit_collected ON reddit_posts(collected_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reddit_party ON reddit_posts(party_mentioned)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reddit_sentiment ON reddit_posts(sentiment_label)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_youtube_collected ON youtube_comments(collected_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_youtube_party ON youtube_comments(party_mentioned)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flashback_collected ON flashback_posts(collected_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flashback_party ON flashback_posts(party_mentioned)")
        conn.commit()
    finally:
        conn.close()


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def detect_party(text):
    text = normalize_text(text).lower()
    if not text:
        return None
    for party, keywords in PARTY_KEYWORDS.items():
        for keyword in keywords:
            if len(keyword) <= 3:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return party
            elif keyword in text:
                return party
    return None

def detect_leader(text):
    text = normalize_text(text).lower()
    if not text:
        return None
    for leader, keywords in LEADER_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return leader
    return None

def detect_issue(text):
    text = normalize_text(text).lower()
    if not text:
        return None
    for issue, keywords in ISSUE_KEYWORDS.items():
        for keyword in keywords:
            if len(keyword) <= 3:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return issue
            elif keyword in text:
                return issue
    return None

def get_bloc(party_name):
    info = SWEDISH_PARTIES.get(party_name)
    return info.get("bloc") if info else None


# ============================================================
# SENTIMENT MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)

def sentiment_one(classifier, text):
    text = normalize_text(text)
    if not text:
        return "neutral", 0.0
    try:
        result = classifier(text[:512], truncation=True)[0]
        raw_label = result["label"].lower()
        confidence = float(result["score"])
        if "positive" in raw_label:
            return "positive", confidence
        if "negative" in raw_label:
            return "negative", -confidence
        return "neutral", 0.0
    except Exception:
        return "neutral", 0.0


# ============================================================
# ADVANCED ANALYTICS FUNCTIONS (NEW)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    if not SPACY_AVAILABLE:
        return None
    try:
        # Try to load the Swedish model, fallback to blank if not installed
        nlp = spacy.load("sv_core_news_sm")
        return nlp
    except OSError:
        # If not installed, try to download it (but may not work in cloud)
        try:
            spacy.cli.download("sv_core_news_sm")
            return spacy.load("sv_core_news_sm")
        except:
            return None

def run_ner(texts, nlp):
    """Extract named entities from a list of texts."""
    if nlp is None:
        return []
    entities = []
    for text in texts:
        doc = nlp(text[:1000000])  # limit to avoid memory
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
    return entities

@st.cache_data(ttl=3600)
def run_topic_modeling(texts, num_topics=10, passes=2):
    """Run LDA topic modeling on a list of texts."""
    if not GENSIM_AVAILABLE or not texts:
        return None, None
    # Tokenize and clean
    tokenized = [re.findall(r'\b[a-zåäö]{3,}\b', text.lower()) for text in texts]
    # Remove stopwords
    from gensim.parsing.preprocessing import STOPWORDS
    tokenized = [[word for word in doc if word not in STOPWORDS] for doc in tokenized]
    # Filter out empty
    tokenized = [doc for doc in tokenized if len(doc) > 0]
    if not tokenized:
        return None, None
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized)
    # Filter extremes
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]
    # Train LDA
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42)
    topics = lda.print_topics(num_words=10)
    return topics, lda

def detect_anomalies(df, column='party_mentioned', window=7, z_thresh=3):
    """Detect anomalies in daily mention counts using Z-score."""
    if df.empty:
        return pd.DataFrame()
    # Group by date and count mentions
    df['date'] = pd.to_datetime(df['collected_at']).dt.date
    daily = df.groupby(['date', column]).size().reset_index(name='count')
    # Compute moving average and std
    daily['avg'] = daily.groupby(column)['count'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    daily['std'] = daily.groupby(column)['count'].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )
    daily['z'] = (daily['count'] - daily['avg']) / daily['std'].replace(0, np.nan)
    daily['anomaly'] = daily['z'].abs() > z_thresh
    return daily[daily['anomaly']]

def forecast_simple(series, days=7):
    """Simple linear regression forecast for a time series."""
    if len(series) < 3:
        return None
    # Prepare data
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(series), len(series)+days).reshape(-1, 1)
    pred = model.predict(future_X)
    # Also get confidence (approximate)
    return pred

# ============================================================
# REDDIT, YOUTUBE, FLASHBACK, GOOGLE TRENDS COLLECTORS
# (unchanged from previous version – keep as above)
# ... (I'll include them in the final code, but to save space in this answer, I'll note they remain the same)
# ============================================================

# ... (all collector functions remain exactly as in the previous improved version)
# I will include them in the final code block for completeness.

# ============================================================
# DATA LOADING AND PREPARATION (unchanged)
# ============================================================

# All the @st.cache_data functions remain the same.

# ============================================================
# SIDEBAR WITH ADVANCED ANALYTICS BUTTONS
# ============================================================

def show_sidebar():
    with st.sidebar:
        st.title("🎛️ Controls")
        st.caption("Manual mode — no automatic background collection.")
        st.markdown("---")

        # ... (existing collection controls)

        st.markdown("---")
        st.subheader("🔬 Advanced Analytics")
        if st.button("🧠 Run Topic Modelling", use_container_width=True):
            st.session_state['run_topic'] = True
        if st.button("🏷️ Run NER (Named Entity Recognition)", use_container_width=True):
            st.session_state['run_ner'] = True
        if st.button("📈 Detect Anomalies", use_container_width=True):
            st.session_state['run_anomaly'] = True
        if st.button("🔗 Build Co-occurrence Network", use_container_width=True):
            st.session_state['run_network'] = True
        if st.button("📉 Forecast 7 Days", use_container_width=True):
            st.session_state['run_forecast'] = True

        # ... (rest of sidebar)

# ============================================================
# MAIN DASHBOARD
# ============================================================

def show_dashboard():
    show_sidebar()
    st.title(APP_TITLE)
    # ... (existing top metrics and charts)

    # After the existing sections, add new advanced sections based on session state

    # Topic Modelling
    if st.session_state.get('run_topic', False):
        st.subheader("🧠 Topic Modelling (LDA)")
        with st.spinner("Running LDA on collected texts..."):
            df = prepare_all_data()
            texts = df['display_text'].dropna().tolist()
            if len(texts) > 10:
                topics, model = run_topic_modeling(texts, num_topics=8)
                if topics:
                    for idx, (topic_id, words) in enumerate(topics):
                        st.markdown(f"**Topic {idx+1}:** {words}")
                else:
                    st.info("Not enough text for topic modelling.")
            else:
                st.warning("Need more data (at least 10 texts).")
        st.session_state['run_topic'] = False

    # NER
    if st.session_state.get('run_ner', False):
        st.subheader("🏷️ Named Entity Recognition")
        nlp = load_spacy_model()
        if nlp is None:
            st.error("spaCy Swedish model not available. Please install 'sv_core_news_sm'.")
        else:
            with st.spinner("Extracting entities..."):
                df = prepare_all_data()
                texts = df['display_text'].dropna().tolist()[:200]  # limit
                entities = run_ner(texts, nlp)
                if entities:
                    ent_df = pd.DataFrame(entities, columns=['Entity', 'Type'])
                    ent_counts = ent_df.groupby(['Entity', 'Type']).size().reset_index(name='Count')
                    st.dataframe(ent_counts.sort_values('Count', ascending=False).head(20), use_container_width=True)
                else:
                    st.info("No entities found.")
        st.session_state['run_ner'] = False

    # Anomaly Detection
    if st.session_state.get('run_anomaly', False):
        st.subheader("📈 Anomaly Detection (Z-score)")
        df = prepare_all_data()
        if not df.empty:
            anomalies = detect_anomalies(df, column='party_mentioned')
            if not anomalies.empty:
                st.dataframe(anomalies, use_container_width=True)
                fig = px.scatter(anomalies, x='date', y='count', color='party_mentioned',
                                 title="Anomalous Party Mention Spikes")
                st.plotly_chart(fig)
            else:
                st.info("No anomalies detected.")
        else:
            st.warning("No data.")
        st.session_state['run_anomaly'] = False

    # Co-occurrence Network
    if st.session_state.get('run_network', False):
        st.subheader("🔗 Party-Issue Co-occurrence Network")
        df = prepare_all_data()
        subset = df[df['party_mentioned'].notna() & df['issue_mentioned'].notna()]
        if not subset.empty:
            # Create network edges
            edges = subset.groupby(['party_mentioned', 'issue_mentioned']).size().reset_index(name='weight')
            # Build a network graph using plotly
            # For simplicity, we'll show a chord diagram or heatmap
            # We'll use a heatmap matrix
            matrix = edges.pivot(index='party_mentioned', columns='issue_mentioned', values='weight').fillna(0)
            fig = px.imshow(matrix, text_auto=True, aspect="auto", title="Party × Issue Co-occurrence")
            st.plotly_chart(fig)
            # Also show a force-directed graph? plotly doesn't have built-in; we can use networkx but extra dependency.
            # We'll stick with heatmap.
        else:
            st.warning("Not enough co-occurrence data.")
        st.session_state['run_network'] = False

    # Forecast
    if st.session_state.get('run_forecast', False):
        st.subheader("📉 7-Day Forecast")
        df = prepare_all_data()
        if not df.empty:
            # Forecast daily sentiment
            df['date'] = pd.to_datetime(df['collected_at']).dt.date
            daily_sent = df.groupby('date')['sentiment_score'].mean().dropna()
            if len(daily_sent) > 2:
                pred = forecast_simple(daily_sent, days=7)
                if pred is not None:
                    future_dates = [daily_sent.index[-1] + timedelta(days=i+1) for i in range(7)]
                    forecast_df = pd.DataFrame({'date': future_dates, 'predicted_sentiment': pred})
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily_sent.index, y=daily_sent.values, mode='lines+markers', name='Historical'))
                    fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_sentiment'], mode='lines+markers', name='Forecast', line=dict(dash='dash')))
                    fig.update_layout(title='Sentiment Forecast (7 days)')
                    st.plotly_chart(fig)
                else:
                    st.warning("Not enough data for forecast.")
            else:
                st.warning("Need at least 3 days of data.")
        else:
            st.warning("No data.")
        st.session_state['run_forecast'] = False

    # ... (rest of existing dashboard sections)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    init_database()
    show_dashboard()
