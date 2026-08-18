import os
import re
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# Optional imports are handled gracefully
try:
    import praw
except ImportError:
    praw = None

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Swedish Election Sentiment 2026",
    page_icon="🇸🇪",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "swedish_election_2026.db"

ELECTION_DATE = datetime(2026, 9, 13)

SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"


# ============================================================
# SWEDISH POLITICAL PARTIES
# ============================================================

SWEDISH_PARTIES = {
    "Socialdemokraterna": {
        "abbr": "S",
        "leader": "Magdalena Andersson",
        "bloc": "Left",
        "keywords": [
            "socialdemokraterna",
            "socialdemokraterna",
            "sossarna",
            "magdalena andersson",
            "socialdemokraterna",
        ],
    },
    "Moderaterna": {
        "abbr": "M",
        "leader": "Ulf Kristersson",
        "bloc": "Right",
        "keywords": [
            "moderaterna",
            "moderaterna",
            "moderaterna",
            "ulf kristersson",
        ],
    },
    "Sverigedemokraterna": {
        "abbr": "SD",
        "leader": "Jimmie Åkesson",
        "bloc": "Right",
        "keywords": [
            "sverigedemokraterna",
            "sverigedemokraterna",
            "sverigedemokraterna",
            "sd",
            "jimmie åkesson",
        ],
    },
    "Centerpartiet": {
        "abbr": "C",
        "leader": "Elisabeth Thand Ringqvist",
        "bloc": "Centre",
        "keywords": [
            "centerpartiet",
            "centerpartiet",
            "centerpartiet",
            "centerpartiet",
        ],
    },
    "Vänsterpartiet": {
        "abbr": "V",
        "leader": "Nooshi Dadgostar",
        "bloc": "Left",
        "keywords": [
            "vänsterpartiet",
            "vansterpartiet",
            "vänstern",
            "nooshi dadgostar",
        ],
    },
    "Kristdemokraterna": {
        "abbr": "KD",
        "leader": "Ebba Busch",
        "bloc": "Right",
        "keywords": [
            "kristdemokraterna",
            "kristdemokraterna",
            "ebba busch",
        ],
    },
    "Liberalerna": {
        "abbr": "L",
        "leader": "Simona Mohamsson",
        "bloc": "Centre",
        "keywords": [
            "liberalerna",
            "liberalerna",
            "folkpartiet",
            "simona mohamsson",
        ],
    },
    "Miljöpartiet": {
        "abbr": "MP",
        "leader": "Amanda Lind",
        "bloc": "Left",
        "keywords": [
            "miljöpartiet",
            "miljopartiet",
            "miljöpartiet de gröna",
            "mp",
            "amanda lind",
        ],
    },
}


# ============================================================
# POLITICAL ISSUES
# ============================================================

ISSUES = {
    "Immigration": [
        "invandring",
        "migration",
        "migranter",
        "flykting",
        "flyktingar",
        "asyl",
        "integration",
        "utvisning",
        "gräns",
        "gränser",
    ],
    "Crime": [
        "brott",
        "brottslighet",
        "kriminalitet",
        "gäng",
        "gängvåld",
        "skjutning",
        "skjutningar",
        "kriminella",
        "fängelse",
        "polis",
        "polisen",
    ],
    "Healthcare": [
        "sjukvård",
        "sjukvard",
        "vård",
        "vard",
        "sjukhus",
        "läkare",
        "läkemedel",
        "1177",
        "vårdkö",
    ],
    "Education": [
        "skola",
        "skolan",
        "skolor",
        "lärare",
        "elever",
        "gymnasium",
        "universitet",
        "utbildning",
    ],
    "Economy": [
        "ekonomi",
        "inflation",
        "ränta",
        "räntor",
        "skatt",
        "skatter",
        "jobb",
        "arbetslöshet",
        "lön",
        "löner",
        "kostnader",
    ],
    "Climate": [
        "klimat",
        "klimatet",
        "utsläpp",
        "miljö",
        "miljön",
        "koldioxid",
        "förnybar",
        "vindkraft",
        "solenergi",
    ],
    "NATO": [
        "nato",
        "försvar",
        "försvaret",
        "militär",
        "militären",
        "säkerhet",
        "ukraina",
        "ryssland",
    ],
    "Housing": [
        "bostad",
        "bostäder",
        "hyra",
        "hyror",
        "bostadsmarknad",
        "lägenhet",
        "lägenheter",
    ],
    "Energy": [
        "el",
        "elpris",
        "elpriser",
        "energi",
        "kärnkraft",
        "kärnkraften",
        "vindkraft",
        "drivmedel",
        "bensin",
        "diesel",
    ],
    "Welfare": [
        "välfärd",
        "bidrag",
        "försäkringskassan",
        "pension",
        "pensioner",
        "socialförsäkring",
    ],
}


# ============================================================
# SEARCH TERMS
# ============================================================

SEARCH_TERMS = [
    "riksdagsval 2026",
    "valet 2026",
    "svensk politik",
    "politik Sverige",
    "regeringen",
    "riksdagen",
    "Socialdemokraterna",
    "Moderaterna",
    "Sverigedemokraterna",
    "Centerpartiet",
    "Vänsterpartiet",
    "Kristdemokraterna",
    "Liberalerna",
    "Miljöpartiet",
    "invandring",
    "integration",
    "kriminalitet",
    "gängvåld",
    "sjukvård",
    "skola",
    "ekonomi",
    "klimat",
    "NATO",
    "bostäder",
    "energi",
    "välfärd",
]


SUBREDDITS = [
    "sweden",
    "svenskpolitik",
    "svenska",
    "europe",
    "worldnews",
    "politics",
]


YOUTUBE_QUERIES = [
    "riksdagsval 2026",
    "svensk politik 2026",
    "valdebatt 2026",
    "Socialdemokraterna 2026",
    "Moderaterna 2026",
    "Sverigedemokraterna 2026",
    "Centerpartiet 2026",
    "Vänsterpartiet 2026",
    "Kristdemokraterna 2026",
    "Liberalerna 2026",
    "Miljöpartiet 2026",
]


GOOGLE_TRENDS_TERMS = [
    "Socialdemokraterna",
    "Moderaterna",
    "Sverigedemokraterna",
    "Centerpartiet",
    "Vänsterpartiet",
    "Kristdemokraterna",
    "Liberalerna",
    "Miljöpartiet",
]


# ============================================================
# SECRETS / ENVIRONMENT VARIABLES
# ============================================================

def get_secret(name, default=None):
    """Read from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name, default)


REDDIT_CLIENT_ID = get_secret("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = get_secret("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = get_secret(
    "REDDIT_USER_AGENT",
    "SwedishElectionSentimentMonitor/1.0"
)

YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY")


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False
    )

    # Helps reduce "database is locked" errors
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    return conn


def init_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reddit_posts (
            id TEXT PRIMARY KEY,
            source TEXT DEFAULT 'reddit',
            subreddit TEXT,
            author TEXT,
            title TEXT,
            text TEXT,
            score INTEGER,
            num_comments INTEGER,
            created_utc REAL,
            url TEXT,
            permalink TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            party_mentioned TEXT,
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
            like_count INTEGER,
            published_at TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            party_mentioned TEXT,
            issue_mentioned TEXT,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_summary (
            date TEXT PRIMARY KEY,
            total_posts INTEGER,
            positive_count INTEGER,
            negative_count INTEGER,
            neutral_count INTEGER,
            avg_sentiment REAL,
            top_positive TEXT,
            top_negative TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS google_trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT,
            date TEXT,
            interest INTEGER,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(keyword, date)
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# PARTY DETECTION
# ============================================================

def detect_party(text):

    if not text:
        return None

    text_lower = text.lower()

    matches = []

    for party, data in SWEDISH_PARTIES.items():

        for keyword in data["keywords"]:

            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

            if re.search(pattern, text_lower):
                matches.append(party)
                break

    if not matches:
        return None

    # If multiple parties are mentioned, keep all
    return ", ".join(matches)


# ============================================================
# ISSUE DETECTION
# ============================================================

def detect_issue(text):

    if not text:
        return None

    text_lower = text.lower()

    matches = []

    for issue, keywords in ISSUES.items():

        for keyword in keywords:

            if keyword.lower() in text_lower:
                matches.append(issue)
                break

    if not matches:
        return None

    return ", ".join(matches)


# ============================================================
# REDDIT
# ============================================================

def get_reddit_client():

    if praw is None:
        st.error(
            "PRAW is not installed. Add `praw` to requirements.txt."
        )
        return None

    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:

        st.warning(
            "Reddit API credentials are missing. "
            "Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
        )

        return None

    try:

        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )

        return reddit

    except Exception as e:

        st.error(f"Could not initialize Reddit: {e}")
        return None


def collect_reddit(limit=300):

    reddit = get_reddit_client()

    if reddit is None:
        return 0

    conn = get_connection()
    cursor = conn.cursor()

    collected = 0

    try:

        for subreddit_name in SUBREDDITS:

            subreddit = reddit.subreddit(subreddit_name)

            for term in SEARCH_TERMS:

                try:

                    posts = subreddit.search(
                        term,
                        sort="new",
                        time_filter="month",
                        limit=max(10, limit // len(SEARCH_TERMS))
                    )

                    for post in posts:

                        title = post.title or ""
                        body = post.selftext or ""
                        full_text = f"{title}\n{body}"

                        party = detect_party(full_text)
                        issue = detect_issue(full_text)

                        cursor.execute("""
                            INSERT OR IGNORE INTO reddit_posts (
                                id,
                                subreddit,
                                author,
                                title,
                                text,
                                score,
                                num_comments,
                                created_utc,
                                url,
                                permalink,
                                party_mentioned,
                                issue_mentioned
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(post.id),
                            subreddit_name,
                            str(post.author) if post.author else "[deleted]",
                            title,
                            body,
                            int(post.score or 0),
                            int(post.num_comments or 0),
                            float(post.created_utc),
                            getattr(post, "url", ""),
                            f"https://reddit.com{post.permalink}",
                            party,
                            issue,
                        ))

                        if cursor.rowcount > 0:
                            collected += 1

                except Exception:
                    continue

        conn.commit()

    except Exception as e:

        st.error(f"Reddit collection error: {e}")

    finally:

        conn.close()

    return collected


# ============================================================
# YOUTUBE
# ============================================================

def get_youtube_client():

    if build is None:

        st.error(
            "Google API client is not installed. "
            "Add `google-api-python-client` to requirements.txt."
        )

        return None

    if not YOUTUBE_API_KEY:

        st.warning(
            "YouTube API key is missing. "
            "Add YOUTUBE_API_KEY to Streamlit secrets."
        )

        return None

    try:

        return build(
            "youtube",
            "v3",
            developerKey=YOUTUBE_API_KEY
        )

    except Exception as e:

        st.error(f"Could not initialize YouTube API: {e}")
        return None


def collect_youtube(max_results=30):

    youtube = get_youtube_client()

    if youtube is None:
        return 0

    conn = get_connection()
    cursor = conn.cursor()

    collected = 0

    try:

        for query in YOUTUBE_QUERIES:

            search_response = youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=min(max_results, 50),
                order="date",
                regionCode="SE",
            ).execute()

            for item in search_response.get("items", []):

                video_id = item["id"].get("videoId")

                if not video_id:
                    continue

                video_title = item["snippet"].get(
                    "title",
                    ""
                )

                try:

                    comments_response = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=100,
                        textFormat="plainText",
                        order="time",
                    ).execute()

                except Exception:
                    continue

                for comment_item in comments_response.get(
                    "items",
                    []
                ):

                    snippet = comment_item["snippet"][
                        "topLevelComment"
                    ]["snippet"]

                    comment_id = comment_item["id"]

                    author = snippet.get(
                        "authorDisplayName",
                        ""
                    )

                    text = snippet.get(
                        "textDisplay",
                        ""
                    )

                   
