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
from wordcloud import WordCloud

# Use an alias to avoid conflict with local "sentiment" module
from transformers import pipeline as transformers_pipeline

# Optional advanced libraries – will be disabled if missing
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

# Use a small English model to save memory (67 MB)
# For better Swedish support, you can switch to:
# "cardiffnlp/twitter-xlm-roberta-base-sentiment" (500 MB)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

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
    conn = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
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
# SENTIMENT MODEL (using alias)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    # Use the aliased pipeline to avoid conflict
    return transformers_pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)

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
# DATABASE RETRY HELPER
# ============================================================

def execute_with_retry(cursor, query, params, retries=5):
    for attempt in range(retries):
        try:
            cursor.execute(query, params)
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                raise


# ============================================================
# ADVANCED ANALYTICS FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    if not SPACY_AVAILABLE:
        return None
    try:
        nlp = spacy.load("sv_core_news_sm")
        return nlp
    except OSError:
        try:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "sv_core_news_sm"], check=True)
            return spacy.load("sv_core_news_sm")
        except:
            return None

def run_ner(texts, nlp):
    if nlp is None:
        return []
    entities = []
    for text in texts:
        doc = nlp(text[:1000000])
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
    return entities

@st.cache_data(ttl=3600)
def run_topic_modeling(texts, num_topics=10, passes=2):
    if not GENSIM_AVAILABLE or not texts:
        return None, None
    tokenized = [re.findall(r'\b[a-zåäö]{3,}\b', text.lower()) for text in texts]
    from gensim.parsing.preprocessing import STOPWORDS
    tokenized = [[word for word in doc if word not in STOPWORDS] for doc in tokenized]
    tokenized = [doc for doc in tokenized if len(doc) > 0]
    if not tokenized:
        return None, None
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42)
    topics = lda.print_topics(num_words=10)
    return topics, lda

def detect_anomalies(df, column='party_mentioned', window=7, z_thresh=3):
    if df.empty:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['collected_at']).dt.date
    daily = df.groupby(['date', column]).size().reset_index(name='count')
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
    if not SKLEARN_AVAILABLE or len(series) < 3:
        return None
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(series), len(series)+days).reshape(-1, 1)
    pred = model.predict(future_X)
    return pred


# ============================================================
# REDDIT COLLECTION
# ============================================================

def collect_reddit(limit=DEFAULT_REDDIT_LIMIT):
    if praw is None:
        return 0, "PRAW is not installed."
    if REDDIT_CLIENT_ID == "YOUR_REDDIT_CLIENT_ID":
        return 0, "Reddit API credentials not configured."
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent="SwedishElectionMonitor/2.0",
        )
    except Exception as e:
        return 0, f"Reddit authentication failed: {e}"

    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        limit_per_term = max(1, int(limit / max(len(SEARCH_TERMS), 1)))
        for subreddit_name in SUBREDDITS:
            try:
                subreddit = reddit.subreddit(subreddit_name)
            except Exception:
                continue
            for term in SEARCH_TERMS:
                try:
                    posts = subreddit.search(term, limit=limit_per_term, sort="new")
                    for post in posts:
                        title = normalize_text(getattr(post, "title", ""))
                        body = normalize_text(getattr(post, "selftext", ""))
                        combined = f"{title} {body}"
                        party = detect_party(combined)
                        leader = detect_leader(combined)
                        issue = detect_issue(combined)
                        author = getattr(post, "author", None)
                        author_name = str(author) if author else "[deleted]"
                        cursor.execute("""
                            INSERT OR IGNORE INTO reddit_posts
                            (id, subreddit, author, title, text, score, num_comments,
                             created_utc, url, permalink, party_mentioned, leader_mentioned, issue_mentioned)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            post.id, subreddit_name, author_name, title, body,
                            int(getattr(post, "score", 0) or 0),
                            int(getattr(post, "num_comments", 0) or 0),
                            float(getattr(post, "created_utc", 0) or 0),
                            getattr(post, "url", ""),
                            getattr(post, "permalink", ""),
                            party, leader, issue
                        ))
                        if cursor.rowcount > 0:
                            count += 1
                except Exception:
                    continue
        conn.commit()
    finally:
        conn.close()
    return count, f"Collected {count} new Reddit posts."


# ============================================================
# YOUTUBE COLLECTION
# ============================================================

def collect_youtube(max_results=DEFAULT_YOUTUBE_RESULTS, comments_per_video=DEFAULT_YOUTUBE_COMMENTS):
    if build is None:
        return 0, "Google API client not installed."
    if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY":
        return 0, "YouTube API key not configured."
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        return 0, f"YouTube API init failed: {e}"

    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    try:
        for query in YOUTUBE_QUERIES:
            try:
                search_response = youtube.search().list(
                    q=query, part="id,snippet", maxResults=max_results, type="video", order="relevance"
                ).execute()
                for video in search_response.get("items", []):
                    video_id = video["id"]["videoId"]
                    video_title = normalize_text(video["snippet"]["title"])
                    try:
                        comments_response = youtube.commentThreads().list(
                            part="snippet", videoId=video_id, maxResults=comments_per_video, order="relevance"
                        ).execute()
                    except Exception:
                        continue
                    for item in comments_response.get("items", []):
                        try:
                            comment = item["snippet"]["topLevelComment"]["snippet"]
                            comment_id = item["id"]
                            text = normalize_text(
                                BeautifulSoup(comment.get("textDisplay", ""), "html.parser").get_text(" ", strip=True)
                            )
                            party = detect_party(text)
                            leader = detect_leader(text)
                            issue = detect_issue(text)
                            cursor.execute("""
                                INSERT OR IGNORE INTO youtube_comments
                                (id, video_id, video_title, author, text, like_count, published_at,
                                 party_mentioned, leader_mentioned, issue_mentioned)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                comment_id, video_id, video_title,
                                comment.get("authorDisplayName", ""), text,
                                int(comment.get("likeCount", 0) or 0),
                                comment.get("publishedAt", ""),
                                party, leader, issue
                            ))
                            if cursor.rowcount > 0:
                                count += 1
                        except Exception:
                            continue
            except Exception:
                continue
        conn.commit()
    finally:
        conn.close()
    return count, f"Collected {count} new YouTube comments."


# ============================================================
# FLASHBACK COLLECTION
# ============================================================

def flashback_search_url(term, page=1):
    return f"{FLASHBACK_BASE_URL}/search.php?fresh&s={requests.utils.quote(term)}&p={page}"

def parse_flashback_search_results(html, base_url=FLASHBACK_BASE_URL):
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    threads = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not re.search(r"/t\d+", href):
            continue
        title = normalize_text(link.get_text())
        if not title or len(title) < 5:
            continue
        full_url = href if href.startswith("http") else base_url + href
        thread_root = re.sub(r"(#.*|&p=\d+)$", "", full_url)
        if thread_root in seen:
            continue
        seen.add(thread_root)
        threads.append((thread_root, title))
    return threads

def parse_flashback_thread(html):
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    for message_div in soup.find_all("div", id=re.compile(r"^post_message_\d+")):
        post_id = message_div["id"].replace("post_message_", "")
        text = normalize_text(message_div.get_text(" ", strip=True))
        if not text:
            continue
        author = ""
        posted_at = ""
        author_tag = soup.find("a", attrs={"data-author-id": True}, href=re.compile(rf"post{post_id}|#post{post_id}"))
        if author_tag:
            author = normalize_text(author_tag.get_text())
        posts.append({"post_id": post_id, "author": author, "text": text, "posted_at": posted_at})
    return posts

def collect_flashback(threads_per_term=DEFAULT_FLASHBACK_THREADS_PER_TERM, posts_per_thread=DEFAULT_FLASHBACK_POSTS_PER_THREAD):
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    errors = []
    proxies = None
    if FLASHBACK_PROXY_URL:
        proxies = {"http": FLASHBACK_PROXY_URL, "https": FLASHBACK_PROXY_URL}
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(FLASHBACK_HEADERS)

    try:
        for term in FLASHBACK_SEARCH_TERMS:
            try:
                search_url = flashback_search_url(term)
                response = session.get(search_url, proxies=proxies, timeout=15)
                if response.status_code != 200:
                    errors.append(f"{term}: HTTP {response.status_code}")
                    continue
                time.sleep(FLASHBACK_REQUEST_DELAY_SECONDS)
                threads = parse_flashback_search_results(response.text)[:threads_per_term]
                for thread_url, thread_title in threads:
                    try:
                        thread_response = session.get(thread_url, proxies=proxies, timeout=15)
                        if thread_response.status_code != 200:
                            continue
                        time.sleep(FLASHBACK_REQUEST_DELAY_SECONDS)
                        thread_id_match = re.search(r"/t(\d+)", thread_url)
                        thread_id = thread_id_match.group(1) if thread_id_match else thread_url
                        posts = parse_flashback_thread(thread_response.text)[:posts_per_thread]
                        for post in posts:
                            post_key = f"fb_{thread_id}_{post['post_id']}"
                            combined_text = f"{thread_title} {post['text']}"
                            party = detect_party(combined_text)
                            leader = detect_leader(combined_text)
                            issue = detect_issue(combined_text)
                            cursor.execute("""
                                INSERT OR IGNORE INTO flashback_posts
                                (id, thread_id, thread_title, thread_url, author, text, post_number, posted_at,
                                 party_mentioned, leader_mentioned, issue_mentioned)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                post_key, thread_id, thread_title, thread_url,
                                post["author"], post["text"], 0, post["posted_at"],
                                party, leader, issue
                            ))
                            if cursor.rowcount > 0:
                                count += 1
                    except Exception as e:
                        errors.append(f"{thread_url}: {e}")
                        continue
            except Exception as e:
                errors.append(f"{term}: {e}")
                continue
        conn.commit()
    finally:
        conn.close()
    if count == 0 and errors:
        return 0, f"Flashback scrape failed: {errors[0]}"
    msg = f"Collected {count} new Flashback posts."
    if errors:
        msg += f" ({len(errors)} requests skipped due to errors.)"
    return count, msg


# ============================================================
# GOOGLE TRENDS COLLECTION
# ============================================================

def collect_google_trends(timeframe=GOOGLE_TRENDS_TIMEFRAME, geo=GOOGLE_TRENDS_GEO):
    if TrendReq is None:
        return 0, "pytrends-modern is not installed."
    terms = list(SWEDISH_PARTIES.keys())
    chunks = [terms[i:i+5] for i in range(0, len(terms), 5)]
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    errors = []
    try:
        trend_client_kwargs = {"hl": "sv-SE", "tz": 60, "retries": 3, "backoff_factor": 0.5}
        if GOOGLE_TRENDS_PROXY_URL:
            trend_client_kwargs["proxies"] = {"http": GOOGLE_TRENDS_PROXY_URL, "https": GOOGLE_TRENDS_PROXY_URL}
        pytrends = TrendReq(**trend_client_kwargs)
        for chunk in chunks:
            try:
                pytrends.build_payload(kw_list=chunk, timeframe=timeframe, geo=geo)
                data = pytrends.interest_over_time()
                if data is None or data.empty:
                    continue
                for date_index, row in data.iterrows():
                    date_str = date_index.strftime("%Y-%m-%d")
                    for term in chunk:
                        if term not in row:
                            continue
                        interest_value = int(row[term])
                        cursor.execute("INSERT OR REPLACE INTO google_trends (date, term, interest) VALUES (?, ?, ?)",
                                       (date_str, term, interest_value))
                        count += 1
                time.sleep(1.5)
            except TooManyRequestsError as e:
                errors.append(f"Rate limited by Google Trends: {e}")
                continue
            except Exception as e:
                errors.append(str(e))
                continue
        conn.commit()
    except Exception as e:
        return 0, f"Google Trends init failed: {e}"
    finally:
        conn.close()
    if count == 0 and errors:
        return 0, f"Google Trends fetch failed: {errors[0]}"
    return count, f"Collected {count} Google Trends data points."


# ============================================================
# COLLECTION RUNS
# ============================================================

def record_collection_run(reddit_count, youtube_count, flashback_count=0):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO collection_runs (reddit_count, youtube_count, flashback_count, total_new)
            VALUES (?, ?, ?, ?)
        """, (reddit_count, youtube_count, flashback_count, reddit_count + youtube_count + flashback_count))
        conn.commit()
    finally:
        conn.close()

def collect_all_data(reddit_limit, youtube_results, youtube_comments):
    reddit_count, reddit_msg = collect_reddit(reddit_limit)
    youtube_count, youtube_msg = collect_youtube(youtube_results, youtube_comments)
    record_collection_run(reddit_count, youtube_count)
    return reddit_count, youtube_count, reddit_msg, youtube_msg


# ============================================================
# SENTIMENT ANALYSIS (with retry)
# ============================================================

def analyze_database():
    classifier = load_sentiment_model()
    conn = get_connection()
    cursor = conn.cursor()
    analyzed = 0
    BATCH = 100

    try:
        # Reddit
        cursor.execute("SELECT id, title, text FROM reddit_posts WHERE sentiment_label IS NULL LIMIT ?", (BATCH,))
        reddit_posts = cursor.fetchall()
        for post_id, title, text in reddit_posts:
            combined = normalize_text(f"{title or ''} {text or ''}")
            label, score = sentiment_one(classifier, combined)
            execute_with_retry(
                cursor,
                "UPDATE reddit_posts SET sentiment_label=?, sentiment_score=? WHERE id=?",
                (label, score, post_id)
            )
            analyzed += 1

        # YouTube
        cursor.execute("SELECT id, text FROM youtube_comments WHERE sentiment_label IS NULL LIMIT ?", (BATCH,))
        youtube_comments = cursor.fetchall()
        for comment_id, text in youtube_comments:
            label, score = sentiment_one(classifier, text)
            execute_with_retry(
                cursor,
                "UPDATE youtube_comments SET sentiment_label=?, sentiment_score=? WHERE id=?",
                (label, score, comment_id)
            )
            analyzed += 1

        # Flashback
        cursor.execute("SELECT id, thread_title, text FROM flashback_posts WHERE sentiment_label IS NULL LIMIT ?", (BATCH,))
        flashback_posts = cursor.fetchall()
        for post_id, thread_title, text in flashback_posts:
            combined = normalize_text(f"{thread_title or ''} {text or ''}")
            label, score = sentiment_one(classifier, combined)
            execute_with_retry(
                cursor,
                "UPDATE flashback_posts SET sentiment_label=?, sentiment_score=? WHERE id=?",
                (label, score, post_id)
            )
            analyzed += 1

        conn.commit()
    finally:
        conn.close()

    if analyzed > 0:
        conn = get_connection()
        try:
            conn.execute("INSERT INTO analysis_runs (items_analyzed) VALUES (?)", (analyzed,))
            conn.commit()
        finally:
            conn.close()
    return analyzed


# ============================================================
# PENDING COUNT
# ============================================================

@st.cache_data(ttl=15)
def count_pending_analysis():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reddit_posts WHERE sentiment_label IS NULL")
        reddit = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM youtube_comments WHERE sentiment_label IS NULL")
        youtube = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM flashback_posts WHERE sentiment_label IS NULL")
        flash = cursor.fetchone()[0]
        return reddit + youtube + flash
    finally:
        conn.close()


# ============================================================
# LOAD DATA (CACHED)
# ============================================================

@st.cache_data(ttl=600)
def load_all_data():
    conn = get_connection()
    try:
        reddit_df = pd.read_sql_query("""
            SELECT id, title, text, score, num_comments, created_utc, subreddit, url, permalink,
                   sentiment_label, sentiment_score, party_mentioned, leader_mentioned, issue_mentioned, collected_at
            FROM reddit_posts
        """, conn)
        youtube_df = pd.read_sql_query("""
            SELECT id, video_id, video_title, text, like_count, published_at,
                   sentiment_label, sentiment_score, party_mentioned, leader_mentioned, issue_mentioned, collected_at
            FROM youtube_comments
        """, conn)
        flashback_df = pd.read_sql_query("""
            SELECT id, thread_id, thread_title, thread_url, author, text, posted_at,
                   sentiment_label, sentiment_score, party_mentioned, leader_mentioned, issue_mentioned, collected_at
            FROM flashback_posts
        """, conn)
    finally:
        conn.close()
    return reddit_df, youtube_df, flashback_df

@st.cache_data(ttl=600)
def load_collection_history():
    conn = get_connection()
    try:
        return pd.read_sql_query("SELECT * FROM collection_runs ORDER BY collected_at DESC LIMIT 30", conn)
    finally:
        conn.close()

@st.cache_data(ttl=21600)
def load_google_trends():
    conn = get_connection()
    try:
        return pd.read_sql_query("SELECT * FROM google_trends ORDER BY date", conn)
    finally:
        conn.close()


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_all_data():
    reddit_df, youtube_df, flashback_df = load_all_data()
    frames = []
    if not reddit_df.empty:
        reddit = reddit_df.copy()
        reddit["source"] = "Reddit"
        reddit["display_text"] = reddit["title"].fillna("") + " " + reddit["text"].fillna("")
        reddit["engagement"] = reddit["score"].fillna(0) + reddit["num_comments"].fillna(0) * 2
        reddit["popularity"] = reddit["score"].fillna(0)
        reddit["title"] = reddit["title"]
        reddit["subreddit"] = reddit["subreddit"]
        reddit["score"] = reddit["score"]
        reddit["num_comments"] = reddit["num_comments"]
        reddit["url"] = reddit["url"]
        reddit["permalink"] = reddit["permalink"]
        frames.append(reddit[["source", "display_text", "sentiment_label", "sentiment_score",
                              "party_mentioned", "leader_mentioned", "issue_mentioned",
                              "engagement", "popularity", "collected_at", "title", "subreddit",
                              "score", "num_comments", "url", "permalink"]])
    if not youtube_df.empty:
        youtube = youtube_df.copy()
        youtube["source"] = "YouTube"
        youtube["display_text"] = youtube["text"].fillna("")
        youtube["engagement"] = youtube["like_count"].fillna(0)
        youtube["popularity"] = youtube["like_count"].fillna(0)
        youtube["title"] = youtube["video_title"]
        youtube["subreddit"] = ""
        youtube["score"] = 0
        youtube["num_comments"] = 0
        youtube["url"] = "https://www.youtube.com/watch?v=" + youtube["video_id"].fillna("")
        youtube["permalink"] = ""
        frames.append(youtube[["source", "display_text", "sentiment_label", "sentiment_score",
                               "party_mentioned", "leader_mentioned", "issue_mentioned",
                               "engagement", "popularity", "collected_at", "title", "subreddit",
                               "score", "num_comments", "url", "permalink"]])
    if not flashback_df.empty:
        flashback = flashback_df.copy()
        flashback["source"] = "Flashback"
        flashback["display_text"] = flashback["thread_title"].fillna("") + " " + flashback["text"].fillna("")
        flashback["engagement"] = 1
        flashback["popularity"] = 0
        flashback["title"] = flashback["thread_title"]
        flashback["subreddit"] = ""
        flashback["score"] = 0
        flashback["num_comments"] = 0
        flashback["url"] = flashback["thread_url"].fillna("")
        flashback["permalink"] = ""
        frames.append(flashback[["source", "display_text", "sentiment_label", "sentiment_score",
                                 "party_mentioned", "leader_mentioned", "issue_mentioned",
                                 "engagement", "popularity", "collected_at", "title", "subreddit",
                                 "score", "num_comments", "url", "permalink"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================
# CORE DASHBOARD HELPERS
# ============================================================

def percentage(part, total):
    if total == 0:
        return 0
    return part / total * 100

def safe_pct_change(current, previous):
    if previous == 0:
        return 100.0 if current > 0 else 0.0
    return (current - previous) / abs(previous) * 100

def get_previous_collection_counts():
    history = load_collection_history()
    if len(history) < 2:
        return None
    return history.iloc[0], history.iloc[1]

def calculate_trends(df):
    if df.empty:
        return {}
    now = datetime.now()
    recent_cutoff = now - timedelta(days=1)
    previous_cutoff = now - timedelta(days=2)
    temp = df.copy()
    temp["collected_dt"] = pd.to_datetime(temp["collected_at"], errors="coerce")
    recent = temp[temp["collected_dt"] >= recent_cutoff]
    previous = temp[(temp["collected_dt"] >= previous_cutoff) & (temp["collected_dt"] < recent_cutoff)]
    results = {}
    recent_issues = recent["issue_mentioned"].dropna().value_counts()
    previous_issues = previous["issue_mentioned"].dropna().value_counts()
    issue_changes = []
    for issue in recent_issues.index:
        current = int(recent_issues.get(issue, 0))
        old = int(previous_issues.get(issue, 0))
        issue_changes.append({"Issue": issue, "Recent": current, "Previous": old,
                              "Change %": safe_pct_change(current, old)})
    results["issues"] = pd.DataFrame(issue_changes).sort_values("Change %", ascending=False) if issue_changes else pd.DataFrame()
    recent_parties = recent["party_mentioned"].dropna().value_counts()
    previous_parties = previous["party_mentioned"].dropna().value_counts()
    party_changes = []
    for party in recent_parties.index:
        current = int(recent_parties.get(party, 0))
        old = int(previous_parties.get(party, 0))
        party_changes.append({"Party": party, "Recent": current, "Previous": old,
                              "Change %": safe_pct_change(current, old)})
    results["parties"] = pd.DataFrame(party_changes).sort_values("Change %", ascending=False) if party_changes else pd.DataFrame()
    return results

def calculate_bloc_summary(analyzed_df):
    party_data = analyzed_df[analyzed_df["party_mentioned"].notna()].copy()
    if party_data.empty:
        return pd.DataFrame()
    party_data["bloc"] = party_data["party_mentioned"].apply(get_bloc)
    party_data = party_data[party_data["bloc"].notna()]
    if party_data.empty:
        return pd.DataFrame()
    summary = party_data.groupby("bloc").agg(
        Mentions=("bloc", "size"),
        Avg_Sentiment=("sentiment_score", "mean"),
        Engagement=("engagement", "sum")
    ).reset_index().rename(columns={"bloc": "Bloc"})
    summary["Bloc"] = summary["Bloc"].str.capitalize()
    return summary.sort_values("Mentions", ascending=False)

STOPWORDS = {"och", "att", "det", "som", "för", "den", "med", "på", "är", "en", "ett", "av", "till", "i", "har",
             "jag", "vi", "de", "dom", "inte", "om", "så", "men", "kan", "var", "the", "and", "for", "that", "this",
             "with", "from", "you", "are", "was", "have", "not", "your", "they", "their", "what", "how", "who", "will",
             "about", "would", "could", "should", "has", "had", "just", "its", "it's", "out", "all", "but", "our", "their", "very"}

def extract_keywords(texts, limit=30):
    counter = Counter()
    for text in texts:
        words = re.findall(r"[A-Za-zÅÄÖåäöÉé\-]{4,}", str(text).lower())
        for word in words:
            if word in STOPWORDS:
                continue
            counter[word] += 1
    return counter.most_common(limit)

def narrative_summary(df):
    if df.empty:
        return []
    results = []
    for issue in ISSUE_KEYWORDS.keys():
        subset = df[df["issue_mentioned"] == issue]
        if subset.empty:
            continue
        total = len(subset)
        positive = len(subset[subset["sentiment_label"] == "positive"])
        negative = len(subset[subset["sentiment_label"] == "negative"])
        neutral = len(subset[subset["sentiment_label"] == "neutral"])
        keywords = extract_keywords(subset["display_text"].tolist(), limit=8)
        results.append({
            "Issue": issue,
            "Mentions": total,
            "Positive %": round(percentage(positive, total), 1),
            "Negative %": round(percentage(negative, total), 1),
            "Neutral %": round(percentage(neutral, total), 1),
            "Top terms": ", ".join(word for word, _ in keywords[:5])
        })
    return results

def generate_alerts(df, trends):
    alerts = []
    if df.empty:
        return alerts
    sentiment_df = df[df["sentiment_label"].notna()]
    if not sentiment_df.empty:
        negative_pct = percentage(len(sentiment_df[sentiment_df["sentiment_label"] == "negative"]), len(sentiment_df))
        if negative_pct >= 60:
            alerts.append(("🔴", "High negative sentiment", f"{negative_pct:.1f}% of conversation is negative."))
    issue_trends = trends.get("issues", pd.DataFrame())
    if not issue_trends.empty:
        for _, row in issue_trends.head(5).iterrows():
            if row["Recent"] < MIN_MENTIONS_FOR_ALERT:
                continue
            if row["Previous"] == 0:
                alerts.append(("🆕", f"New activity: {row['Issue']}", f"{int(row['Recent'])} mentions with no prior baseline."))
            elif row["Change %"] >= 50:
                alerts.append(("🟠", f"Issue spike: {row['Issue']}", f"Mentions rose {row['Change %']:.0f}% ({int(row['Previous'])} → {int(row['Recent'])})."))
    party_trends = trends.get("parties", pd.DataFrame())
    if not party_trends.empty:
        for _, row in party_trends.head(5).iterrows():
            if row["Recent"] < MIN_MENTIONS_FOR_ALERT:
                continue
            if row["Previous"] == 0:
                alerts.append(("🆕", f"New activity: {row['Party']}", f"{int(row['Recent'])} mentions with no prior baseline."))
            elif row["Change %"] >= 50:
                alerts.append(("🟠", f"Attention spike: {row['Party']}", f"Mentions rose {row['Change %']:.0f}% ({int(row['Previous'])} → {int(row['Recent'])})."))
    return alerts

def generate_intelligence_brief(df, trends, bloc_summary=None):
    if df.empty:
        return "No analyzed data is available yet. Collect data and run sentiment analysis first."
    analyzed = df[df["sentiment_label"].notna()].copy()
    if analyzed.empty:
        return "Data has been collected but not analyzed yet."
    total = len(analyzed)
    positive = len(analyzed[analyzed["sentiment_label"] == "positive"])
    negative = len(analyzed[analyzed["sentiment_label"] == "negative"])
    neutral = len(analyzed[analyzed["sentiment_label"] == "neutral"])
    issue_counts = analyzed["issue_mentioned"].dropna().value_counts()
    party_counts = analyzed["party_mentioned"].dropna().value_counts()
    leader_counts = analyzed["leader_mentioned"].dropna().value_counts()
    top_issue = issue_counts.index[0] if len(issue_counts) else "No dominant issue detected"
    top_party = party_counts.index[0] if len(party_counts) else "No dominant party detected"
    top_leader = leader_counts.index[0] if len(leader_counts) else "No dominant leader detected"
    brief = []
    brief.append("# Campaign Intelligence Brief")
    brief.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    brief.append("")
    brief.append("## Overall conversation")
    brief.append(f"- Analyzed items: **{total:,}**")
    brief.append(f"- Positive: **{percentage(positive, total):.1f}%**")
    brief.append(f"- Negative: **{percentage(negative, total):.1f}%**")
    brief.append(f"- Neutral: **{percentage(neutral, total):.1f}%**")
    brief.append("")
    brief.append("## Attention")
    brief.append(f"- Most-mentioned issue: **{top_issue}**")
    brief.append(f"- Most-mentioned party: **{top_party}**")
    brief.append(f"- Most-mentioned leader: **{top_leader}**")
    brief.append("")
    if bloc_summary is not None and not bloc_summary.empty:
        brief.append("## Bloc-level conversation")
        for _, row in bloc_summary.iterrows():
            brief.append(f"- {row['Bloc']}: {int(row['Mentions']):,} mentions, avg sentiment {row['Avg_Sentiment']:.3f}")
        brief.append("")
    if not issue_counts.empty:
        brief.append("## Leading issues")
        for issue, count in issue_counts.head(5).items():
            brief.append(f"- {issue}: {count:,} mentions")
    if not party_counts.empty:
        brief.append("")
        brief.append("## Party attention")
        for party, count in party_counts.head(8).items():
            brief.append(f"- {party}: {count:,} mentions")
    issue_trends = trends.get("issues", pd.DataFrame())
    if not issue_trends.empty:
        brief.append("")
        brief.append("## Recent changes")
        for _, row in issue_trends.head(5).iterrows():
            brief.append(f"- {row['Issue']}: {row['Change %']:+.0f}% change ({int(row['Recent'])} recent, {int(row['Previous'])} previous)")
    brief.append("")
    brief.append("## Interpretation")
    brief.append("This report describes online conversation captured from the connected sources. It should not be treated as a representative poll of the Swedish electorate.")
    brief.append("")
    brief.append("Use large changes, engagement spikes and recurring narratives as subjects for further investigation rather than as direct measures of voter preference.")
    return "\n".join(brief)


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():
    with st.sidebar:
        st.title("🎛️ Controls")
        st.caption("Manual mode — no automatic background collection.")
        st.markdown("---")

        st.subheader("📥 Collection")
        reddit_limit = st.slider("Reddit posts", min_value=30, max_value=200, value=DEFAULT_REDDIT_LIMIT, step=10)
        youtube_results = st.slider("YouTube videos/query", min_value=5, max_value=25, value=DEFAULT_YOUTUBE_RESULTS, step=5)
        youtube_comments = st.slider("Comments/video", min_value=10, max_value=50, value=DEFAULT_YOUTUBE_COMMENTS, step=10)

        if st.button("🔄 Collect New Data", use_container_width=True, type="primary"):
            with st.spinner("Collecting Reddit and YouTube data..."):
                reddit_count, youtube_count, reddit_msg, youtube_msg = collect_all_data(
                    reddit_limit, youtube_results, youtube_comments
                )
            load_all_data.clear()
            load_collection_history.clear()
            count_pending_analysis.clear()
            st.success(f"Reddit: {reddit_count} new | YouTube: {youtube_count} new")
            st.caption(reddit_msg)
            st.caption(youtube_msg)

        if st.button("🧠 Analyze New Data", use_container_width=True):
            with st.spinner("Running sentiment analysis..."):
                analyzed = analyze_database()
            load_all_data.clear()
            count_pending_analysis.clear()
            st.success(f"Analyzed {analyzed} new items.")

        pending = count_pending_analysis()
        if pending > 0:
            st.caption(f"⏳ {pending:,} items awaiting sentiment analysis (100 per click).")

        st.markdown("---")
        st.subheader("📈 Google Trends")
        if GOOGLE_TRENDS_PROXY_URL:
            st.caption("🟢 Proxy configured.")
        else:
            st.caption("⚪ No proxy (may be rate-limited).")
        if st.button("📈 Fetch Google Trends", use_container_width=True):
            with st.spinner("Fetching Google Trends data..."):
                trends_count, trends_msg = collect_google_trends()
            load_google_trends.clear()
            if trends_count > 0:
                st.success(trends_msg)
            else:
                st.error(trends_msg)

        st.markdown("---")
        st.subheader("🗨️ Flashback")
        if FLASHBACK_PROXY_URL:
            st.caption("🟢 Proxy configured.")
        else:
            st.caption("⚪ No proxy – be gentle with requests.")
        flashback_threads = st.slider("Threads per term", 1, 5, DEFAULT_FLASHBACK_THREADS_PER_TERM, step=1)
        flashback_posts = st.slider("Posts per thread", 5, 30, DEFAULT_FLASHBACK_POSTS_PER_THREAD, step=5)
        if st.button("🗨️ Fetch Flashback Posts", use_container_width=True):
            with st.spinner("Scraping Flashback..."):
                fb_count, fb_msg = collect_flashback(flashback_threads, flashback_posts)
                record_collection_run(0, 0, fb_count)
            load_all_data.clear()
            load_collection_history.clear()
            count_pending_analysis.clear()
            if fb_count > 0:
                st.success(fb_msg)
            else:
                st.error(fb_msg)

        st.markdown("---")
        if st.button("📊 Refresh Dashboard", use_container_width=True):
            load_all_data.clear()
            load_collection_history.clear()
            load_google_trends.clear()
            count_pending_analysis.clear()
            st.rerun()

        # Clear data
        if st.button("🗑️ Clear All Data", use_container_width=True):
            if st.checkbox("Confirm deletion of all data?"):
                conn = get_connection()
                try:
                    conn.execute("DELETE FROM reddit_posts")
                    conn.execute("DELETE FROM youtube_comments")
                    conn.execute("DELETE FROM flashback_posts")
                    conn.execute("DELETE FROM google_trends")
                    conn.execute("DELETE FROM collection_runs")
                    conn.execute("DELETE FROM analysis_runs")
                    conn.commit()
                    st.success("All data cleared.")
                    load_all_data.clear()
                    load_collection_history.clear()
                    load_google_trends.clear()
                    count_pending_analysis.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing data: {e}")
                finally:
                    conn.close()

        st.markdown("---")
        st.subheader("🔬 Advanced Analytics")
        advanced_status = []
        if not GENSIM_AVAILABLE:
            advanced_status.append("Topic modelling disabled (gensim missing)")
        if not SPACY_AVAILABLE:
            advanced_status.append("NER disabled (spacy missing)")
        if not SKLEARN_AVAILABLE:
            advanced_status.append("Forecast disabled (scikit-learn missing)")
        if advanced_status:
            st.caption("⚠️ " + " | ".join(advanced_status))

        if st.button("🧠 Run Topic Modelling", use_container_width=True):
            st.session_state['run_topic'] = True
        if st.button("🏷️ Run NER", use_container_width=True):
            st.session_state['run_ner'] = True
        if st.button("📈 Detect Anomalies", use_container_width=True):
            st.session_state['run_anomaly'] = True
        if st.button("🔗 Build Co-occurrence Network", use_container_width=True):
            st.session_state['run_network'] = True
        if st.button("📉 Forecast 7 Days", use_container_width=True):
            st.session_state['run_forecast'] = True

        st.markdown("---")
        st.subheader("🧠 Model")
        st.code(MODEL_NAME, language="text")
        st.markdown("---")
        st.subheader("🇸🇪 Party reference")
        for party, info in SWEDISH_PARTIES.items():
            st.markdown(f"**{info['abbrev']}** — {party} _({info['bloc']})_")


# ============================================================
# MAIN DASHBOARD
# ============================================================

def show_dashboard():
    show_sidebar()
    st.title(APP_TITLE)
    days_until = (ELECTION_DATE - datetime.now()).days
    if days_until >= 0:
        st.markdown(f"**Riksdagsval: 13 September 2026** | **{days_until} days remaining**")
    else:
        st.markdown("**Riksdagsval: 13 September 2026**")
    st.caption("Online political conversation monitoring from Reddit, YouTube, Flashback, and Google Trends. Manual collection.")

    df = prepare_all_data()
    if df.empty:
        st.info("No data collected yet. Use the sidebar to collect.")
        return

    analyzed = df[df["sentiment_label"].notna()].copy()

    # Metrics
    st.subheader("📊 Executive Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Items", f"{len(df):,}")
    with col2:
        st.metric("Analyzed", f"{len(analyzed):,}")
    if not analyzed.empty:
        positive_pct = percentage(len(analyzed[analyzed["sentiment_label"] == "positive"]), len(analyzed))
        negative_pct = percentage(len(analyzed[analyzed["sentiment_label"] == "negative"]), len(analyzed))
        avg_sentiment = analyzed["sentiment_score"].mean()
    else:
        positive_pct = negative_pct = 0
        avg_sentiment = 0
    with col3:
        st.metric("Positive", f"{positive_pct:.1f}%")
    with col4:
        st.metric("Negative", f"{negative_pct:.1f}%")
    with col5:
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    with col6:
        st.metric("Engagement", f"{int(df['engagement'].sum()):,}")

    if analyzed.empty:
        st.warning("Data collected but not analyzed. Run sentiment analysis in sidebar.")
        return

    trends = calculate_trends(df)
    bloc_summary = calculate_bloc_summary(analyzed)

    # Alerts
    st.subheader("🚨 Intelligence Alerts")
    alerts = generate_alerts(analyzed, trends)
    if alerts:
        for icon, title, message in alerts:
            if icon == "🆕":
                st.info(f"{icon} **{title}** — {message}")
            else:
                st.warning(f"{icon} **{title}** — {message}")
    else:
        st.success("No major spikes detected.")

    # Sentiment
    st.subheader("🗣️ Public Conversation Sentiment")
    col_left, col_right = st.columns(2)
    with col_left:
        sentiment_counts = analyzed["sentiment_label"].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     title="Sentiment Distribution",
                     color=sentiment_counts.index,
                     color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"})
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        source_sentiment = analyzed.groupby(["source", "sentiment_label"]).size().reset_index(name="count")
        fig = px.bar(source_sentiment, x="source", y="count", color="sentiment_label",
                     title="Sentiment by Source",
                     color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"})
        st.plotly_chart(fig, use_container_width=True)

    # Bloc
    st.subheader("⚖️ Bloc Intelligence (Left / Right / Center)")
    if not bloc_summary.empty:
        bloc_col1, bloc_col2 = st.columns(2)
        with bloc_col1:
            st.dataframe(bloc_summary, use_container_width=True, hide_index=True)
        with bloc_col2:
            fig = px.bar(bloc_summary, x="Bloc", y="Mentions", color="Bloc",
                         title="Conversation Volume by Bloc", color_discrete_map=BLOC_COLORS)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Bloc assignments are a simplification – C has shifted across recent elections.")
    else:
        st.info("Not enough party-tagged data yet for bloc view.")

    # Trending
    st.subheader("🔥 What's Trending")
    trend_col1, trend_col2 = st.columns(2)
    with trend_col1:
        issue_trends = trends.get("issues", pd.DataFrame())
        if not issue_trends.empty:
            st.markdown("**Issues showing largest recent changes**")
            st.dataframe(issue_trends.head(10), use_container_width=True, hide_index=True)
            st.caption("Previous=0 means new activity, not growth.")
        else:
            st.info("More data needed for trends.")
    with trend_col2:
        party_trends = trends.get("parties", pd.DataFrame())
        if not party_trends.empty:
            st.markdown("**Parties showing largest recent changes**")
            st.dataframe(party_trends.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("More data needed for trends.")

    # Party Intelligence
    st.subheader("🏛️ Party Intelligence")
    party_data = analyzed[analyzed["party_mentioned"].notna()].copy()
    if not party_data.empty:
        party_summary = party_data.groupby("party_mentioned").agg(
            Mentions=("party_mentioned", "size"),
            Avg_Sentiment=("sentiment_score", "mean"),
            Engagement=("engagement", "sum")
        ).reset_index().sort_values("Mentions", ascending=False)
        st.dataframe(party_summary, use_container_width=True, hide_index=True)
        fig = px.bar(party_data.groupby(["party_mentioned", "sentiment_label"]).size().reset_index(name="count"),
                     x="party_mentioned", y="count", color="sentiment_label",
                     title="Party Mentions by Sentiment",
                     color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No party mentions detected.")

    # Leader
    st.subheader("👤 Leader Attention")
    leader_data = analyzed[analyzed["leader_mentioned"].notna()]
    if not leader_data.empty:
        leader_summary = leader_data.groupby("leader_mentioned").agg(
            Mentions=("leader_mentioned", "size"),
            Avg_Sentiment=("sentiment_score", "mean"),
            Engagement=("engagement", "sum")
        ).reset_index().sort_values("Mentions", ascending=False)
        st.dataframe(leader_summary, use_container_width=True, hide_index=True)
    else:
        st.info("No leader names detected yet.")

    # Party × Issue
    st.subheader("🧩 Party × Issue Matrix")
    matrix_data = analyzed[analyzed["party_mentioned"].notna() & analyzed["issue_mentioned"].notna()]
    if not matrix_data.empty:
        matrix = pd.crosstab(matrix_data["party_mentioned"], matrix_data["issue_mentioned"])
        fig = px.imshow(matrix, text_auto=True, aspect="auto", title="Conversation Volume by Party and Issue")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(matrix, use_container_width=True)
    else:
        st.info("Not enough party + issue data yet.")

    # Top Issues
    st.subheader("🎯 Top Issues")
    issue_data = analyzed[analyzed["issue_mentioned"].notna()]
    if not issue_data.empty:
        issue_counts = issue_data["issue_mentioned"].value_counts().head(10)
        fig = px.bar(x=issue_counts.index, y=issue_counts.values, labels={"x": "Issue", "y": "Mentions"},
                     title="Most Discussed Issues")
        st.plotly_chart(fig, use_container_width=True)

    # Google Trends
    st.subheader("📈 Google Search Interest")
    st.caption("Google Trends interest-over-time (Sweden). Values are relative 0–100.")
    trends_df = load_google_trends()
    if not trends_df.empty:
        trends_df = trends_df.copy()
        trends_df["date"] = pd.to_datetime(trends_df["date"])
        fig = px.line(trends_df, x="date", y="interest", color="term",
                      title="Search Interest Over Time by Party (Sweden)")
        st.plotly_chart(fig, use_container_width=True)
        latest_date = trends_df["date"].max()
        latest_snapshot = trends_df[trends_df["date"] == latest_date].sort_values("interest", ascending=False)
        st.markdown(f"**Latest snapshot ({latest_date.strftime('%Y-%m-%d')})**")
        st.dataframe(latest_snapshot[["term", "interest"]].rename(columns={"term": "Party", "interest": "Search Interest"}),
                     use_container_width=True, hide_index=True)
        if not party_data.empty:
            mention_counts = party_data["party_mentioned"].value_counts().rename_axis("term").reset_index(name="mentions")
            comparison = latest_snapshot[["term", "interest"]].merge(mention_counts, on="term", how="left")
            comparison["mentions"] = comparison["mentions"].fillna(0)
            st.markdown("**Conversation mentions vs. search interest**")
            st.dataframe(comparison.rename(columns={"term": "Party", "interest": "Search Interest", "mentions": "Conversation Mentions"}),
                         use_container_width=True, hide_index=True)
    else:
        st.info("No Google Trends data yet. Use sidebar to fetch.")

    # Engagement
    st.subheader("🔥 Popularity & Engagement")
    eng_left, eng_right = st.columns(2)
    with eng_left:
        st.markdown("### Reddit — Most Upvoted")
        reddit_popular = df[(df["source"] == "Reddit") & (df["score"] > 0)].sort_values("score", ascending=False).head(10)
        if not reddit_popular.empty:
            st.dataframe(reddit_popular[["score", "num_comments", "party_mentioned", "issue_mentioned", "sentiment_label", "title"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No Reddit popularity data yet.")
    with eng_right:
        st.markdown("### YouTube — Most Liked Comments")
        youtube_popular = df[(df["source"] == "YouTube") & (df["popularity"] > 0)].sort_values("popularity", ascending=False).head(10)
        if not youtube_popular.empty:
            display = youtube_popular[["popularity", "party_mentioned", "issue_mentioned", "sentiment_label", "title", "display_text"]].copy()
            display = display.rename(columns={"popularity": "Likes", "title": "Video", "display_text": "Comment"})
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("No YouTube like data yet.")

    # Top Engagement
    st.markdown("### 🏆 Highest Engagement")
    top_eng = df.sort_values("engagement", ascending=False).head(15)
    st.dataframe(top_eng[["source", "engagement", "popularity", "party_mentioned", "issue_mentioned", "sentiment_label", "title", "display_text"]],
                 use_container_width=True, hide_index=True)

    # Narrative
    st.subheader("📰 Narrative Monitor")
    narratives = narrative_summary(analyzed)
    if narratives:
        narrative_df = pd.DataFrame(narratives)
        st.dataframe(narrative_df, use_container_width=True, hide_index=True)
        st.caption("Narratives are inferred from recurring issue keywords. They are monitoring signals, not polling results.")
    else:
        st.info("No narratives detected yet.")

    # Keywords
    st.subheader("🔎 Conversation Keywords")
    keywords = extract_keywords(analyzed["display_text"].tolist(), limit=40)
    if keywords:
        keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Count"])
        st.dataframe(keyword_df, use_container_width=True, hide_index=True)

    # Sentiment Trend
    st.subheader("📈 Sentiment Trend")
    trend_df = analyzed.copy()
    trend_df["date"] = pd.to_datetime(trend_df["collected_at"], errors="coerce").dt.date
    daily = trend_df.groupby("date").agg(Avg_Sentiment=("sentiment_score", "mean"), Items=("sentiment_score", "count")).reset_index()
    if not daily.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["Avg_Sentiment"], mode="lines+markers", name="Average Sentiment"))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(xaxis_title="Date", yaxis_title="Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    # Word Clouds
    st.subheader("☁️ Conversation Word Clouds")
    wc_left, wc_right = st.columns(2)
    with wc_left:
        st.markdown("### Negative conversation")
        neg_text = " ".join(analyzed[analyzed["sentiment_label"] == "negative"]["display_text"].dropna().astype(str))
        if neg_text.strip():
            wc = WordCloud(width=800, height=450, background_color="white", colormap="Reds").generate(neg_text)
            st.image(wc.to_array(), use_container_width=True)
        else:
            st.info("No negative text.")
    with wc_right:
        st.markdown("### Positive conversation")
        pos_text = " ".join(analyzed[analyzed["sentiment_label"] == "positive"]["display_text"].dropna().astype(str))
        if pos_text.strip():
            wc = WordCloud(width=800, height=450, background_color="white", colormap="Greens").generate(pos_text)
            st.image(wc.to_array(), use_container_width=True)
        else:
            st.info("No positive text.")

    # Collection History
    st.subheader("🕐 Collection History")
    history = load_collection_history()
    if not history.empty:
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("No collection history yet.")

    # Intelligence Brief
    st.subheader("📄 Campaign Intelligence Brief")
    brief = generate_intelligence_brief(df, trends, bloc_summary)
    st.text(brief)

    # ============================================================
    # ADVANCED ANALYTICS SECTIONS (triggered by sidebar buttons)
    # ============================================================

    # Topic Modelling
    if st.session_state.get('run_topic', False):
        st.subheader("🧠 Topic Modelling (LDA)")
        if not GENSIM_AVAILABLE:
            st.error("Topic modelling is disabled because 'gensim' is not installed.")
        else:
            with st.spinner("Running LDA on collected texts..."):
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
        if not SPACY_AVAILABLE:
            st.error("NER is disabled because 'spacy' is not installed.")
        else:
            nlp = load_spacy_model()
            if nlp is None:
                st.error("Swedish spaCy model could not be loaded. Please install 'sv_core_news_sm'.")
            else:
                with st.spinner("Extracting entities..."):
                    texts = df['display_text'].dropna().tolist()[:200]
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

    # Co-occurrence Network (heatmap)
    if st.session_state.get('run_network', False):
        st.subheader("🔗 Party-Issue Co-occurrence Network")
        subset = df[df['party_mentioned'].notna() & df['issue_mentioned'].notna()]
        if not subset.empty:
            edges = subset.groupby(['party_mentioned', 'issue_mentioned']).size().reset_index(name='weight')
            matrix = edges.pivot(index='party_mentioned', columns='issue_mentioned', values='weight').fillna(0)
            fig = px.imshow(matrix, text_auto=True, aspect="auto", title="Party × Issue Co-occurrence")
            st.plotly_chart(fig)
        else:
            st.warning("Not enough co-occurrence data.")
        st.session_state['run_network'] = False

    # Forecast
    if st.session_state.get('run_forecast', False):
        st.subheader("📉 7-Day Forecast")
        if not SKLEARN_AVAILABLE:
            st.error("Forecast is disabled because 'scikit-learn' is not installed.")
        else:
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
        st.session_state['run_forecast'] = False


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    init_database()
    show_dashboard()
