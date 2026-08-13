import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud
from transformers import pipeline

# Optional APIs
try:
    import praw
except ImportError:
    praw = None

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None


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

# Keep this manual and lightweight for Streamlit free tier.
DEFAULT_REDDIT_LIMIT = 120
DEFAULT_YOUTUBE_RESULTS = 15
DEFAULT_YOUTUBE_COMMENTS = 30

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


# ============================================================
# PARTY DATA
# ============================================================

SWEDISH_PARTIES = {
    "Socialdemokraterna": {
        "leader": "Magdalena Andersson",
        "abbrev": "S",
        "bloc": "left",
    },
    "Moderaterna": {
        "leader": "Ulf Kristersson",
        "abbrev": "M",
        "bloc": "right",
    },
    "Sverigedemokraterna": {
        "leader": "Jimmie Åkesson",
        "abbrev": "SD",
        "bloc": "right",
    },
    "Kristdemokraterna": {
        "leader": "Ebba Busch",
        "abbrev": "KD",
        "bloc": "right",
    },
    "Liberalerna": {
        "leader": "Johan Pehrson",
        "abbrev": "L",
        "bloc": "right",
    },
    "Centerpartiet": {
        "leader": "Muharrem Demirok",
        "abbrev": "C",
        "bloc": "center",
    },
    "Miljöpartiet": {
        "leader": "Amanda Lind",
        "abbrev": "MP",
        "bloc": "left",
    },
    "Vänsterpartiet": {
        "leader": "Nooshi Dadgostar",
        "abbrev": "V",
        "bloc": "left",
    },
}


PARTY_KEYWORDS = {
    "Socialdemokraterna": [
        "socialdemokraterna",
        "socialdemokrat",
        "sosse",
        "magdalena andersson",
        "s-partiet",
    ],
    "Moderaterna": [
        "moderaterna",
        "moderat",
        "ulf kristersson",
        "m-partiet",
    ],
    "Sverigedemokraterna": [
        "sverigedemokraterna",
        "sverigedemokrat",
        "jimmie akesson",
        "jimmie åkesson",
        "sd",
    ],
    "Kristdemokraterna": [
        "kristdemokraterna",
        "kristdemokrat",
        "ebba busch",
        "kd",
    ],
    "Liberalerna": [
        "liberalerna",
        "folkpartiet",
        "johan pehrson",
        "liberal",
        "fp",
    ],
    "Centerpartiet": [
        "centerpartiet",
        "centerparti",
        "muharrem demirok",
        "c-partiet",
    ],
    "Miljöpartiet": [
        "miljöpartiet",
        "miljopartiet",
        "miljöparti",
        "miljoparti",
        "amanda lind",
        "mp",
    ],
    "Vänsterpartiet": [
        "vänsterpartiet",
        "vansterpartiet",
        "vänsterparti",
        "vansterparti",
        "nooshi dadgostar",
        "v-partiet",
    ],
}


LEADER_KEYWORDS = {
    "Magdalena Andersson": [
        "magdalena andersson",
    ],
    "Ulf Kristersson": [
        "ulf kristersson",
    ],
    "Jimmie Åkesson": [
        "jimmie åkesson",
        "jimmie akesson",
    ],
    "Ebba Busch": [
        "ebba busch",
    ],
    "Johan Pehrson": [
        "johan pehrson",
    ],
    "Muharrem Demirok": [
        "muharrem demirok",
    ],
    "Amanda Lind": [
        "amanda lind",
    ],
    "Nooshi Dadgostar": [
        "nooshi dadgostar",
    ],
}


ISSUE_KEYWORDS = {
    "Immigration": [
        "invandring",
        "immigration",
        "migrant",
        "migranter",
        "flykting",
        "flyktingar",
        "asyl",
        "integration",
    ],
    "Crime": [
        "kriminalitet",
        "brott",
        "brottslighet",
        "crime",
        "våld",
        "vald",
        "skjutning",
        "skjutningar",
        "gäng",
        "gang",
        "gängen",
    ],
    "Healthcare": [
        "sjukvård",
        "sjukvard",
        "vård",
        "vard",
        "healthcare",
        "sjukhus",
        "läkare",
        "lakare",
    ],
    "Education": [
        "skola",
        "skolan",
        "utbildning",
        "school",
        "lärare",
        "larare",
        "elever",
    ],
    "Economy": [
        "ekonomi",
        "economy",
        "inflation",
        "priser",
        "pris",
        "bnp",
        "recession",
        "ränta",
        "ranta",
        "skatt",
        "skatter",
    ],
    "Climate": [
        "klimat",
        "climate",
        "miljö",
        "miljo",
        "koldioxid",
        "utsläpp",
        "utslapp",
    ],
    "NATO & Defence": [
        "nato",
        "försvar",
        "forsvar",
        "defense",
        "militär",
        "militar",
        "värnplikt",
        "varnplikt",
    ],
    "Housing": [
        "bostad",
        "bostäder",
        "bostader",
        "housing",
        "hyra",
        "hyror",
        "bostadsbrist",
    ],
    "Energy": [
        "elpris",
        "elpriser",
        "energi",
        "energy",
        "el",
        "kärnkraft",
        "karnkraft",
        "vindkraft",
    ],
    "Welfare": [
        "bidrag",
        "welfare",
        "försörjningsstöd",
        "forsorjningsstod",
        "socialbidrag",
        "pension",
        "välfärd",
        "valfard",
    ],
}


SEARCH_TERMS = [
    "riksdagsval 2026",
    "val 2026",
    "Swedish election 2026",
    "Sverige val",
    "valrörelse 2026",
    "Socialdemokraterna",
    "Moderaterna",
    "Sverigedemokraterna",
    "Kristdemokraterna",
    "Liberalerna",
    "Centerpartiet",
    "Miljöpartiet",
    "Vänsterpartiet",
    "Magdalena Andersson",
    "Ulf Kristersson",
    "Jimmie Åkesson",
    "Ebba Busch",
    "invandring",
    "kriminalitet",
    "sjukvård",
    "skola",
    "ekonomi",
    "klimat",
    "NATO",
    "bidrag",
    "bostad",
    "elpris",
    "försvar",
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
    "Magdalena Andersson",
    "Ulf Kristersson",
    "Jimmie Åkesson",
    "Sverigedemokraterna",
]


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


REDDIT_CLIENT_ID = get_secret(
    "REDDIT_CLIENT_ID",
    "YOUR_REDDIT_CLIENT_ID",
)

REDDIT_CLIENT_SECRET = get_secret(
    "REDDIT_CLIENT_SECRET",
    "YOUR_REDDIT_CLIENT_SECRET",
)

YOUTUBE_API_KEY = get_secret(
    "YOUTUBE_API_KEY",
    "YOUR_YOUTUBE_API_KEY",
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    """
    Creates a short-lived SQLite connection.

    WAL + busy_timeout greatly reduces the database-lock problem
    that can happen on Streamlit when the app reruns.
    """
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
    )

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")

    return conn


def init_database():
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
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
            """
        )

        cursor.execute(
            """
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
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reddit_count INTEGER DEFAULT 0,
                youtube_count INTEGER DEFAULT 0,
                total_new INTEGER DEFAULT 0
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                items_analyzed INTEGER DEFAULT 0
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment_summary (
                date TEXT PRIMARY KEY,
                total_posts INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                avg_sentiment REAL
            )
            """
        )

        # Add columns if an older database exists.
        add_column_if_missing(
            cursor,
            "reddit_posts",
            "leader_mentioned",
            "TEXT",
        )

        add_column_if_missing(
            cursor,
            "youtube_comments",
            "leader_mentioned",
            "TEXT",
        )

        conn.commit()

    finally:
        conn.close()


def add_column_if_missing(cursor, table, column, column_type):
    try:
        cursor.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
        )
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


# ============================================================
# DATABASE RETRY
# ============================================================

def execute_with_retry(function, retries=5):
    """
    Runs a database operation with retry logic.
    Helps prevent temporary SQLite locking failures.
    """

    last_error = None

    for attempt in range(retries):
        try:
            return function()

        except sqlite3.OperationalError as e:
            last_error = e

            if "locked" not in str(e).lower():
                raise

            time.sleep(0.5 * (attempt + 1))

    raise last_error


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

            # Avoid matching tiny abbreviations like "sd"
            # inside unrelated words.
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


# ============================================================
# SENTIMENT MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )


def sentiment_one(classifier, text):

    text = normalize_text(text)

    if not text:
        return "neutral", 0.0

    try:

        result = classifier(
            text[:512],
            truncation=True,
        )[0]

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
# REDDIT COLLECTION
# ============================================================

def collect_reddit(limit=DEFAULT_REDDIT_LIMIT):

    if praw is None:
        return 0, "PRAW is not installed."

    if REDDIT_CLIENT_ID == "YOUR_REDDIT_CLIENT_ID":
        return (
            0,
            "Reddit API credentials are not configured.",
        )

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

        limit_per_term = max(
            1,
            int(limit / max(len(SEARCH_TERMS), 1)),
        )

        for subreddit_name in SUBREDDITS:

            try:
                subreddit = reddit.subreddit(
                    subreddit_name
                )
            except Exception:
                continue

            for term in SEARCH_TERMS:

                try:

                    posts = subreddit.search(
                        term,
                        limit=limit_per_term,
                        sort="new",
                    )

                    for post in posts:

                        title = normalize_text(
                            getattr(post, "title", "")
                        )

                        body = normalize_text(
                            getattr(post, "selftext", "")
                        )

                        combined = f"{title} {body}"

                        party = detect_party(combined)
                        leader = detect_leader(combined)
                        issue = detect_issue(combined)

                        author = getattr(
                            post,
                            "author",
                            None,
                        )

                        author_name = (
                            str(author)
                            if author
                            else "[deleted]"
                        )

                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO reddit_posts
                            (
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
                                leader_mentioned,
                                issue_mentioned
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                post.id,
                                subreddit_name,
                                author_name,
                                title,
                                body,
                                int(
                                    getattr(
                                        post,
                                        "score",
                                        0,
                                    )
                                    or 0
                                ),
                                int(
                                    getattr(
                                        post,
                                        "num_comments",
                                        0,
                                    )
                                    or 0
                                ),
                                float(
                                    getattr(
                                        post,
                                        "created_utc",
                                        0,
                                    )
                                    or 0
                                ),
                                getattr(
                                    post,
                                    "url",
                                    "",
                                ),
                                getattr(
                                    post,
                                    "permalink",
                                    "",
                                ),
                                party,
                                leader,
                                issue,
                            ),
                        )

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

def collect_youtube(
    max_results=DEFAULT_YOUTUBE_RESULTS,
    comments_per_video=DEFAULT_YOUTUBE_COMMENTS,
):

    if build is None:
        return 0, "Google API client is not installed."

    if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY":
        return (
            0,
            "YouTube API key is not configured.",
        )

    try:

        youtube = build(
            "youtube",
            "v3",
            developerKey=YOUTUBE_API_KEY,
        )

    except Exception as e:

        return 0, f"YouTube API initialization failed: {e}"

    conn = get_connection()
    cursor = conn.cursor()

    count = 0

    try:

        for query in YOUTUBE_QUERIES:

            try:

                search_response = (
                    youtube.search()
                    .list(
                        q=query,
                        part="id,snippet",
                        maxResults=max_results,
                        type="video",
                        order="relevance",
                    )
                    .execute()
                )

                for video in search_response.get(
                    "items",
                    [],
                ):

                    video_id = video["id"]["videoId"]

                    video_title = normalize_text(
                        video["snippet"]["title"]
                    )

                    try:

                        comments_response = (
                            youtube.commentThreads()
                            .list(
                                part="snippet",
                                videoId=video_id,
                                maxResults=comments_per_video,
                                order="relevance",
                            )
                            .execute()
                        )

                    except Exception:
                        continue

                    for item in comments_response.get(
                        "items",
                        [],
                    ):

                        try:

                            comment = item[
                                "snippet"
                            ][
                                "topLevelComment"
                            ][
                                "snippet"
                            ]

                            comment_id = item["id"]

                            text = normalize_text(
                                BeautifulSoup(
                                    comment.get(
                                        "textDisplay",
                                        "",
                                    ),
                                    "html.parser",
                                ).get_text(
                                    " ",
                                    strip=True,
                                )
                            )

                            party = detect_party(text)
                            leader = detect_leader(text)
                            issue = detect_issue(text)

                            cursor.execute(
                                """
                                INSERT OR IGNORE INTO youtube_comments
                                (
                                    id,
                                    video_id,
                                    video_title,
                                    author,
                                    text,
                                    like_count,
                                    published_at,
                                    party_mentioned,
                                    leader_mentioned,
                                    issue_mentioned
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    comment_id,
                                    video_id,
                                    video_title,
                                    comment.get(
                                        "authorDisplayName",
                                        "",
                                    ),
                                    text,
                                    int(
                                        comment.get(
                                            "likeCount",
                                            0,
                                        )
                                        or 0
                                    ),
                                    comment.get(
                                        "publishedAt",
                                        "",
                                    ),
                                    party,
                                    leader,
                                    issue,
                                ),
                            )

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
# COLLECTION RUN
# ============================================================

def record_collection_run(
    reddit_count,
    youtube_count,
):

    conn = get_connection()

    try:

        conn.execute(
            """
            INSERT INTO collection_runs
            (reddit_count, youtube_count, total_new)
            VALUES (?, ?, ?)
            """,
            (
                reddit_count,
                youtube_count,
                reddit_count + youtube_count,
            ),
        )

        conn.commit()

    finally:
        conn.close()


def collect_all_data(
    reddit_limit,
    youtube_results,
    youtube_comments,
):

    reddit_count, reddit_msg = collect_reddit(
        reddit_limit
    )

    youtube_count, youtube_msg = collect_youtube(
        youtube_results,
        youtube_comments,
    )

    record_collection_run(
        reddit_count,
        youtube_count,
    )

    return (
        reddit_count,
        youtube_count,
        reddit_msg,
        youtube_msg,
    )


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_database():

    classifier = load_sentiment_model()

    conn = get_connection()
    cursor = conn.cursor()

    analyzed = 0

    try:

        # -------------------------
        # REDDIT
        # -------------------------

        cursor.execute(
            """
            SELECT id, title, text
            FROM reddit_posts
            WHERE sentiment_label IS NULL
            LIMIT 300
            """
        )

        reddit_posts = cursor.fetchall()

        for post_id, title, text in reddit_posts:

            combined = normalize_text(
                f"{title or ''} {text or ''}"
            )

            label, score = sentiment_one(
                classifier,
                combined,
            )

            cursor.execute(
                """
                UPDATE reddit_posts
                SET sentiment_label=?,
                    sentiment_score=?
                WHERE id=?
                """,
                (
                    label,
                    score,
                    post_id,
                ),
            )

            analyzed += 1

        # -------------------------
        # YOUTUBE
        # -------------------------

        cursor.execute(
            """
            SELECT id, text
            FROM youtube_comments
            WHERE sentiment_label IS NULL
            LIMIT 300
            """
        )

        youtube_comments = cursor.fetchall()

        for comment_id, text in youtube_comments:

            label, score = sentiment_one(
                classifier,
                text,
            )

            cursor.execute(
                """
                UPDATE youtube_comments
                SET sentiment_label=?,
                    sentiment_score=?
                WHERE id=?
                """,
                (
                    label,
                    score,
                    comment_id,
                ),
            )

            analyzed += 1

        conn.commit()

    finally:
        conn.close()

    if analyzed > 0:

        conn = get_connection()

        try:

            conn.execute(
                """
                INSERT INTO analysis_runs
                (items_analyzed)
                VALUES (?)
                """,
                (analyzed,),
            )

            conn.commit()

        finally:
            conn.close()

    return analyzed


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=30)
def load_all_data():

    conn = get_connection()

    try:

        reddit_df = pd.read_sql_query(
            """
            SELECT
                id,
                title,
                text,
                score,
                num_comments,
                created_utc,
                subreddit,
                url,
                permalink,
                sentiment_label,
                sentiment_score,
                party_mentioned,
                leader_mentioned,
                issue_mentioned,
                collected_at
            FROM reddit_posts
            """,
            conn,
        )

        youtube_df = pd.read_sql_query(
            """
            SELECT
                id,
                video_id,
                video_title,
                text,
                like_count,
                published_at,
                sentiment_label,
                sentiment_score,
                party_mentioned,
                leader_mentioned,
                issue_mentioned,
                collected_at
            FROM youtube_comments
            """,
            conn,
        )

    finally:
        conn.close()

    return reddit_df, youtube_df


@st.cache_data(ttl=30)
def load_collection_history():

    conn = get_connection()

    try:

        return pd.read_sql_query(
            """
            SELECT *
            FROM collection_runs
            ORDER BY collected_at DESC
            LIMIT 30
            """,
            conn,
        )

    finally:
        conn.close()


def prepare_all_data():

    reddit_df, youtube_df = load_all_data()

    frames = []

    if not reddit_df.empty:

        reddit = reddit_df.copy()

        reddit["source"] = "Reddit"

        reddit["display_text"] = (
            reddit["title"].fillna("")
            + " "
            + reddit["text"].fillna("")
        )

        reddit["engagement"] = (
            reddit["score"].fillna(0)
            + reddit["num_comments"].fillna(0) * 2
        )

        reddit["popularity"] = reddit["score"].fillna(0)

        frames.append(
            reddit[
                [
                    "source",
                    "display_text",
                    "sentiment_label",
                    "sentiment_score",
                    "party_mentioned",
                    "leader_mentioned",
                    "issue_mentioned",
                    "engagement",
                    "popularity",
                    "collected_at",
                    "title",
                    "subreddit",
                    "score",
                    "num_comments",
                    "url",
                    "permalink",
                ]
            ]
        )

    if not youtube_df.empty:

        youtube = youtube_df.copy()

        youtube["source"] = "YouTube"

        youtube["display_text"] = (
            youtube["text"].fillna("")
        )

        youtube["engagement"] = (
            youtube["like_count"].fillna(0)
        )

        youtube["popularity"] = (
            youtube["like_count"].fillna(0)
        )

        youtube["title"] = youtube[
            "video_title"
        ]

        youtube["subreddit"] = ""

        youtube["score"] = 0

        youtube["num_comments"] = 0

        youtube["url"] = (
            "https://www.youtube.com/watch?v="
            + youtube["video_id"].fillna("")
        )

        youtube["permalink"] = ""

        frames.append(
            youtube[
                [
                    "source",
                    "display_text",
                    "sentiment_label",
                    "sentiment_score",
                    "party_mentioned",
                    "leader_mentioned",
                    "issue_mentioned",
                    "engagement",
                    "popularity",
                    "collected_at",
                    "title",
                    "subreddit",
                    "score",
                    "num_comments",
                    "url",
                    "permalink",
                ]
            ]
        )

    if not frames:

        return pd.DataFrame()

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    return df


# ============================================================
# SUMMARY HELPERS
# ============================================================

def percentage(part, total):

    if total == 0:
        return 0

    return part / total * 100


def safe_pct_change(current, previous):

    if previous == 0:

        if current > 0:
            return 100.0

        return 0.0

    return (
        (current - previous)
        / abs(previous)
        * 100
    )


def get_previous_collection_counts():

    history = load_collection_history()

    if len(history) < 2:
        return None

    latest = history.iloc[0]

    previous = history.iloc[1]

    return latest, previous


# ============================================================
# TREND ANALYSIS
# ============================================================

def calculate_trends(df):

    if df.empty:
        return {}

    now = datetime.now()

    # Compare recent 24h against the preceding period.
    recent_cutoff = now - timedelta(days=1)
    previous_cutoff = now - timedelta(days=2)

    temp = df.copy()

    temp["collected_dt"] = pd.to_datetime(
        temp["collected_at"],
        errors="coerce",
    )

    recent = temp[
        temp["collected_dt"] >= recent_cutoff
    ]

    previous = temp[
        (temp["collected_dt"] >= previous_cutoff)
        & (temp["collected_dt"] < recent_cutoff)
    ]

    results = {}

    # Issues
    recent_issues = (
        recent["issue_mentioned"]
        .dropna()
        .value_counts()
    )

    previous_issues = (
        previous["issue_mentioned"]
        .dropna()
        .value_counts()
    )

    issue_changes = []

    for issue in recent_issues.index:

        current = int(recent_issues.get(issue, 0))
        old = int(previous_issues.get(issue, 0))

        issue_changes.append(
            {
                "Issue": issue,
                "Recent": current,
                "Previous": old,
                "Change %": safe_pct_change(
                    current,
                    old,
                ),
            }
        )

    results["issues"] = pd.DataFrame(
        issue_changes
    ).sort_values(
        "Change %",
        ascending=False,
    ) if issue_changes else pd.DataFrame()

    # Parties
    recent_parties = (
        recent["party_mentioned"]
        .dropna()
        .value_counts()
    )

    previous_parties = (
        previous["party_mentioned"]
        .dropna()
        .value_counts()
    )

    party_changes = []

    for party in recent_parties.index:

        current = int(recent_parties.get(party, 0))
        old = int(previous_parties.get(party, 0))

        party_changes.append(
            {
                "Party": party,
                "Recent": current,
                "Previous": old,
                "Change %": safe_pct_change(
                    current,
                    old,
                ),
            }
        )

    results["parties"] = pd.DataFrame(
        party_changes
    ).sort_values(
        "Change %",
        ascending=False,
    ) if party_changes else pd.DataFrame()

    return results


# ============================================================
# KEYWORD / NARRATIVE ANALYSIS
# ============================================================

STOPWORDS = {
    "och",
    "att",
    "det",
    "som",
    "för",
    "den",
    "med",
    "på",
    "är",
    "en",
    "ett",
    "av",
    "till",
    "i",
    "har",
    "jag",
    "vi",
    "de",
    "dom",
    "inte",
    "om",
    "så",
    "men",
    "kan",
    "var",
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "you",
    "are",
    "was",
    "have",
    "not",
    "your",
    "they",
    "their",
    "what",
    "how",
    "who",
    "will",
    "about",
    "would",
    "could",
    "should",
    "has",
    "had",
    "just",
    "its",
    "it's",
    "out",
    "all",
    "but",
    "our",
    "their",
    "very",
}


def extract_keywords(texts, limit=30):

    counter = Counter()

    for text in texts:

        words = re.findall(
            r"[A-Za-zÅÄÖåäöÉé\-]{4,}",
            str(text).lower(),
        )

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

        subset = df[
            df["issue_mentioned"] == issue
        ]

        if subset.empty:
            continue

        total = len(subset)

        positive = len(
            subset[
                subset["sentiment_label"]
                == "positive"
            ]
        )

        negative = len(
            subset[
                subset["sentiment_label"]
                == "negative"
            ]
        )

        neutral = len(
            subset[
                subset["sentiment_label"]
                == "neutral"
            ]
        )

        keywords = extract_keywords(
            subset["display_text"].tolist(),
            limit=8,
        )

        results.append(
            {
                "Issue": issue,
                "Mentions": total,
                "Positive %": round(
                    percentage(
                        positive,
                        total,
                    ),
                    1,
                ),
                "Negative %": round(
                    percentage(
                        negative,
                        total,
                    ),
                    1,
                ),
                "Neutral %": round(
                    percentage(
                        neutral,
                        total,
                    ),
                    1,
                ),
                "Top terms": ", ".join(
                    word
                    for word, _ in keywords[:5]
                ),
            }
        )

    return results


# ============================================================
# ALERTS
# ============================================================

def generate_alerts(df, trends):

    alerts = []

    if df.empty:
        return alerts

    # Overall negative sentiment.
    sentiment_df = df[
        df["sentiment_label"].notna()
    ]

    if not sentiment_df.empty:

        negative_pct = percentage(
            len(
                sentiment_df[
                    sentiment_df[
                        "sentiment_label"
                    ]
                    == "negative"
                ]
            ),
            len(sentiment_df),
        )

        if negative_pct >= 60:

            alerts.append(
                (
                    "🔴",
                    "High negative sentiment",
                    f"{negative_pct:.1f}% of analyzed conversation "
                    "is currently negative.",
                )
            )

    # Issue spikes.
    issue_trends = trends.get(
        "issues",
        pd.DataFrame(),
    )

    if not issue_trends.empty:

        for _, row in issue_trends.head(3).iterrows():

            if (
                row["Change %"] >= 50
                and row["Recent"] >= 5
            ):

                alerts.append(
                    (
                        "🟠",
                        f"Issue spike: {row['Issue']}",
                        f"Mentions increased approximately "
                        f"{row['Change %']:.0f}%.",
                    )
                )

    # Party spikes.
    party_trends = trends.get(
        "parties",
        pd.DataFrame(),
    )

    if not party_trends.empty:

        for _, row in party_trends.head(3).iterrows():

            if (
                row["Change %"] >= 50
                and row["Recent"] >= 5
            ):

                alerts.append(
                    (
                        "🟠",
                        f"Attention spike: {row['Party']}",
                        f"Mentions increased approximately "
                        f"{row['Change %']:.0f}%.",
                    )
                )

    return alerts


# ============================================================
# CAMPAIGN INTELLIGENCE BRIEF
# ============================================================

def generate_intelligence_brief(df, trends):

    if df.empty:

        return (
            "No analyzed data is available yet. "
            "Collect data and run sentiment analysis first."
        )

    analyzed = df[
        df["sentiment_label"].notna()
    ].copy()

    if analyzed.empty:

        return (
            "Data has been collected but not analyzed yet."
        )

    total = len(analyzed)

    positive = len(
        analyzed[
            analyzed["sentiment_label"]
            == "positive"
        ]
    )

    negative = len(
        analyzed[
            analyzed["sentiment_label"]
            == "negative"
        ]
    )

    neutral = len(
        analyzed[
            analyzed["sentiment_label"]
            == "neutral"
        ]
    )

    issue_counts = (
        analyzed[
            "issue_mentioned"
        ]
        .dropna()
        .value_counts()
    )

    party_counts = (
        analyzed[
            "party_mentioned"
        ]
        .dropna()
        .value_counts()
    )

    leader_counts = (
        analyzed[
            "leader_mentioned"
        ]
        .dropna()
        .value_counts()
    )

    top_issue = (
        issue_counts.index[0]
        if len(issue_counts)
        else "No dominant issue detected"
    )

    top_party = (
        party_counts.index[0]
        if len(party_counts)
        else "No dominant party detected"
    )

    top_leader = (
        leader_counts.index[0]
        if len(leader_counts)
        else "No dominant leader detected"
    )

    brief = []

    brief.append(
        "# Campaign Intelligence Brief"
    )

    brief.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    brief.append("")

    brief.append("## Overall conversation")

    brief.append(
        f"- Analyzed items: **{total:,}**"
    )

    brief.append(
        f"- Positive: **{percentage(positive, total):.1f}%**"
    )

    brief.append(
        f"- Negative: **{percentage(negative, total):.1f}%**"
    )

    brief.append(
        f"- Neutral: **{percentage(neutral, total):.1f}%**"
    )

    brief.append("")

    brief.append("## Attention")

    brief.append(
        f"- Most-mentioned issue: **{top_issue}**"
    )

    brief.append(
        f"- Most-mentioned party: **{top_party}**"
    )

    brief.append(
        f"- Most-mentioned leader: **{top_leader}**"
    )

    brief.append("")

    if not issue_counts.empty:

        brief.append("## Leading issues")

        for issue, count in issue_counts.head(5).items():

            brief.append(
                f"- {issue}: {count:,} mentions"
            )

    if not party_counts.empty:

        brief.append("")

        brief.append("## Party attention")

        for party, count in party_counts.head(8).items():

            brief.append(
                f"- {party}: {count:,} mentions"
            )

    # Trending issues.
    issue_trends = trends.get(
        "issues",
        pd.DataFrame(),
    )

    if not issue_trends.empty:

        brief.append("")

        brief.append("## Recent changes")

        for _, row in issue_trends.head(5).iterrows():

            brief.append(
                f"- {row['Issue']}: "
                f"{row['Change %']:+.0f}% change "
                f"({int(row['Recent'])} recent mentions)"
            )

    brief.append("")

    brief.append("## Interpretation")

    brief.append(
        "This report describes online conversation "
        "captured from the connected sources. It should "
        "not be treated as a representative poll of the "
        "Swedish electorate."
    )

    brief.append("")

    brief.append(
        "Use large changes, engagement spikes and "
        "recurring narratives as subjects for further "
        "investigation rather than as direct measures "
        "of voter preference."
    )

    return "\n".join(brief)


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.title("🎛️ Controls")

        st.caption(
            "Manual mode — no automatic background collection."
        )

        st.markdown("---")

        st.subheader("📥 Collection")

        reddit_limit = st.slider(
            "Reddit posts",
            min_value=30,
            max_value=200,
            value=DEFAULT_REDDIT_LIMIT,
            step=10,
        )

        youtube_results = st.slider(
            "YouTube videos/query",
            min_value=5,
            max_value=25,
            value=DEFAULT_YOUTUBE_RESULTS,
            step=5,
        )

        youtube_comments = st.slider(
            "Comments/video",
            min_value=10,
            max_value=50,
            value=DEFAULT_YOUTUBE_COMMENTS,
            step=10,
        )

        if st.button(
            "🔄 Collect New Data",
            use_container_width=True,
            type="primary",
        ):

            with st.spinner(
                "Collecting Reddit and YouTube data..."
            ):

                (
                    reddit_count,
                    youtube_count,
                    reddit_msg,
                    youtube_msg,
                ) = collect_all_data(
                    reddit_limit,
                    youtube_results,
                    youtube_comments,
                )

            load_all_data.clear()
            load_collection_history.clear()

            st.success(
                f"Reddit: {reddit_count} new | "
                f"YouTube: {youtube_count} new"
            )

            st.caption(reddit_msg)
            st.caption(youtube_msg)

        if st.button(
            "🧠 Analyze New Data",
            use_container_width=True,
        ):

            with st.spinner(
                "Running sentiment analysis..."
            ):

                analyzed = analyze_database()

            load_all_data.clear()

            st.success(
                f"Analyzed {analyzed} new items."
            )

        if st.button(
            "📊 Refresh Dashboard",
            use_container_width=True,
        ):

            load_all_data.clear()
            load_collection_history.clear()

            st.rerun()

        st.markdown("---")

        st.subheader("🧠 Model")

        st.caption(
            "Sentiment model:"
        )

        st.code(
            MODEL_NAME,
            language="text",
        )

        st.markdown("---")

        st.subheader("🇸🇪 Party reference")

        for party, info in SWEDISH_PARTIES.items():

            st.markdown(
                f"**{info['abbrev']}** — {party}"
            )


# ============================================================
# MAIN DASHBOARD
# ============================================================

def show_dashboard():

    show_sidebar()

    st.title(APP_TITLE)

    days_until = (
        ELECTION_DATE - datetime.now()
    ).days

    if days_until >= 0:

        st.markdown(
            f"**Riksdagsval: 13 September 2026** "
            f"| **{days_until} days remaining**"
        )

    else:

        st.markdown(
            "**Riksdagsval: 13 September 2026**"
        )

    st.caption(
        "Online political conversation monitoring from Reddit and YouTube. "
        "Manual collection is used to keep the app lightweight."
    )

    df = prepare_all_data()

    if df.empty:

        st.info(
            "No data has been collected yet. "
            "Use **Collect New Data** in the sidebar."
        )

        return

    analyzed = df[
        df["sentiment_label"].notna()
    ].copy()

    # ========================================================
    # TOP METRICS
    # ========================================================

    st.subheader("📊 Executive Overview")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "Total Items",
            f"{len(df):,}",
        )

    with col2:
        st.metric(
            "Analyzed",
            f"{len(analyzed):,}",
        )

    if not analyzed.empty:

        positive_pct = percentage(
            len(
                analyzed[
                    analyzed["sentiment_label"]
                    == "positive"
                ]
            ),
            len(analyzed),
        )

        negative_pct = percentage(
            len(
                analyzed[
                    analyzed["sentiment_label"]
                    == "negative"
                ]
            ),
            len(analyzed),
        )

        avg_sentiment = (
            analyzed["sentiment_score"]
            .mean()
        )

    else:

        positive_pct = 0
        negative_pct = 0
        avg_sentiment = 0

    with col3:
        st.metric(
            "Positive",
            f"{positive_pct:.1f}%",
        )

    with col4:
        st.metric(
            "Negative",
            f"{negative_pct:.1f}%",
        )

    with col5:
        st.metric(
            "Avg Sentiment",
            f"{avg_sentiment:.3f}",
        )

    with col6:
        st.metric(
            "Engagement",
            f"{int(df['engagement'].sum()):,}",
        )

    if analyzed.empty:

        st.warning(
            "Data has been collected, but sentiment "
            "has not been analyzed yet."
        )

        return

    # ========================================================
    # TREND ANALYSIS
    # ========================================================

    trends = calculate_trends(df)

    # ========================================================
    # ALERTS
    # ========================================================

    st.subheader("🚨 Intelligence Alerts")

    alerts = generate_alerts(
        analyzed,
        trends,
    )

    if alerts:

        for icon, title, message in alerts:

            st.warning(
                f"{icon} **{title}** — {message}"
            )

    else:

        st.success(
            "No major automatically detected spikes "
            "under the current thresholds."
        )

    # ========================================================
    # SENTIMENT
    # ========================================================

    st.subheader("🗣️ Public Conversation Sentiment")

    col_left, col_right = st.columns(2)

    with col_left:

        sentiment_counts = (
            analyzed[
                "sentiment_label"
            ]
            .value_counts()
        )

        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col_right:

        source_sentiment = (
            analyzed.groupby(
                [
                    "source",
                    "sentiment_label",
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        fig = px.bar(
            source_sentiment,
            x="source",
            y="count",
            color="sentiment_label",
            title="Sentiment by Source",
            color_discrete_map={
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # TRENDING
    # ========================================================

    st.subheader("🔥 What's Trending")

    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:

        issue_trends = trends.get(
            "issues",
            pd.DataFrame(),
        )

        if not issue_trends.empty:

            st.markdown(
                "**Issues showing the largest recent changes**"
            )

            st.dataframe(
                issue_trends.head(10),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "More historical collection runs are "
                "needed to calculate changes."
            )

    with trend_col2:

        party_trends = trends.get(
            "parties",
            pd.DataFrame(),
        )

        if not party_trends.empty:

            st.markdown(
                "**Parties showing the largest recent changes**"
            )

            st.dataframe(
                party_trends.head(10),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "More historical collection data is needed."
            )

    # ========================================================
    # PARTY INTELLIGENCE
    # ========================================================

    st.subheader("🏛️ Party Intelligence")

    party_data = analyzed[
        analyzed["party_mentioned"].notna()
    ].copy()

    if not party_data.empty:

        party_summary = (
            party_data.groupby(
                "party_mentioned"
            )
            .agg(
                Mentions=("party_mentioned", "size"),
                Avg_Sentiment=(
                    "sentiment_score",
                    "mean",
                ),
                Engagement=(
                    "engagement",
                    "sum",
                ),
            )
            .reset_index()
            .sort_values(
                "Mentions",
                ascending=False,
            )
        )

        st.dataframe(
            party_summary,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            party_data.groupby(
                [
                    "party_mentioned",
                    "sentiment_label",
                ]
            )
            .size()
            .reset_index(name="count"),
            x="party_mentioned",
            y="count",
            color="sentiment_label",
            title="Party Mentions by Sentiment",
            color_discrete_map={
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.info(
            "No party mentions detected."
        )

    # ========================================================
    # LEADER INTELLIGENCE
    # ========================================================

    st.subheader("👤 Leader Attention")

    leader_data = analyzed[
        analyzed["leader_mentioned"].notna()
    ]

    if not leader_data.empty:

        leader_summary = (
            leader_data.groupby(
                "leader_mentioned"
            )
            .agg(
                Mentions=("leader_mentioned", "size"),
                Avg_Sentiment=(
                    "sentiment_score",
                    "mean",
                ),
                Engagement=(
                    "engagement",
                    "sum",
                ),
            )
            .reset_index()
            .sort_values(
                "Mentions",
                ascending=False,
            )
        )

        st.dataframe(
            leader_summary,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No leader names detected yet."
        )

    # ========================================================
    # PARTY × ISSUE
    # ========================================================

    st.subheader("🧩 Party × Issue Matrix")

    matrix_data = analyzed[
        analyzed["party_mentioned"].notna()
        & analyzed["issue_mentioned"].notna()
    ]

    if not matrix_data.empty:

        matrix = pd.crosstab(
            matrix_data["party_mentioned"],
            matrix_data["issue_mentioned"],
        )

        fig = px.imshow(
            matrix,
            text_auto=True,
            aspect="auto",
            title="Conversation Volume by Party and Issue",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.dataframe(
            matrix,
            use_container_width=True,
        )

    else:

        st.info(
            "Not enough party + issue data yet."
        )

    # ========================================================
    # ISSUES
    # ========================================================

    st.subheader("🎯 Top Issues")

    issue_data = analyzed[
        analyzed["issue_mentioned"].notna()
    ]

    if not issue_data.empty:

        issue_counts = (
            issue_data[
                "issue_mentioned"
            ]
            .value_counts()
            .head(10)
        )

        fig = px.bar(
            x=issue_counts.index,
            y=issue_counts.values,
            labels={
                "x": "Issue",
                "y": "Mentions",
            },
            title="Most Discussed Issues",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # ENGAGEMENT
    # ========================================================

    st.subheader("🔥 Popularity & Engagement")

    engagement_left, engagement_right = st.columns(2)

    with engagement_left:

        st.markdown(
            "### Reddit — Most Upvoted"
        )

        reddit_popular = df[
            (df["source"] == "Reddit")
            & (df["score"] > 0)
        ].sort_values(
            "score",
            ascending=False,
        ).head(10)

        if not reddit_popular.empty:

            display = reddit_popular[
                [
                    "score",
                    "num_comments",
                    "party_mentioned",
                    "issue_mentioned",
                    "sentiment_label",
                    "title",
                ]
            ].copy()

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No Reddit popularity data yet."
            )

    with engagement_right:

        st.markdown(
            "### YouTube — Most Liked Comments"
        )

        youtube_popular = df[
            (df["source"] == "YouTube")
            & (df["popularity"] > 0)
        ].sort_values(
            "popularity",
            ascending=False,
        ).head(10)

        if not youtube_popular.empty:

            display = youtube_popular[
                [
                    "popularity",
                    "party_mentioned",
                    "issue_mentioned",
                    "sentiment_label",
                    "title",
                    "display_text",
                ]
            ].copy()

            display = display.rename(
                columns={
                    "popularity": "Likes",
                    "title": "Video",
                    "display_text": "Comment",
                }
            )

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No YouTube like data yet."
            )

    # ========================================================
    # TOP ENGAGEMENT
    # ========================================================

    st.markdown(
        "### 🏆 Highest Engagement"
    )

    top_engagement = df.sort_values(
        "engagement",
        ascending=False,
    ).head(15)

    st.dataframe(
        top_engagement[
            [
                "source",
                "engagement",
                "popularity",
                "party_mentioned",
                "issue_mentioned",
                "sentiment_label",
                "title",
                "display_text",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # NARRATIVE MONITOR
    # ========================================================

    st.subheader("📰 Narrative Monitor")

    narratives = narrative_summary(
        analyzed
    )

    if narratives:

        narrative_df = pd.DataFrame(
            narratives
        )

        st.dataframe(
            narrative_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Narratives are inferred from recurring issue "
            "keywords and associated conversation terms. "
            "They are monitoring signals, not representative "
            "polling results."
        )

    else:

        st.info(
            "No narratives detected yet."
        )

    # ========================================================
    # KEYWORDS
    # ========================================================

    st.subheader("🔎 Conversation Keywords")

    keywords = extract_keywords(
        analyzed[
            "display_text"
        ].tolist(),
        limit=40,
    )

    if keywords:

        keyword_df = pd.DataFrame(
            keywords,
            columns=[
                "Keyword",
                "Count",
            ],
        )

        st.dataframe(
            keyword_df,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # SENTIMENT TREND
    # ========================================================

    st.subheader("📈 Sentiment Trend")

    trend_df = analyzed.copy()

    trend_df["date"] = pd.to_datetime(
        trend_df["collected_at"],
        errors="coerce",
    ).dt.date

    daily = (
        trend_df.groupby("date")
        .agg(
            Avg_Sentiment=(
                "sentiment_score",
                "mean",
            ),
            Items=(
                "sentiment_score",
                "count",
            ),
        )
        .reset_index()
    )

    if not daily.empty:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["Avg_Sentiment"],
                mode="lines+markers",
                name="Average Sentiment",
            )
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # WORD CLOUDS
    # ========================================================

    st.subheader("☁️ Conversation Word Clouds")

    wc_left, wc_right = st.columns(2)

    with wc_left:

        st.markdown(
            "### Negative conversation"
        )

        negative_text = " ".join(
            analyzed[
                analyzed["sentiment_label"]
                == "negative"
            ]["display_text"]
            .dropna()
            .astype(str)
        )

        if negative_text.strip():

            wc = WordCloud(
                width=800,
                height=450,
                background_color="white",
                colormap="Reds",
            ).generate(
                negative_text
            )

            st.image(
                wc.to_array(),
                use_container_width=True,
            )

        else:

            st.info(
                "No negative text available."
            )

    with wc_right:

        st.markdown(
            "### Positive conversation"
        )

        positive_text = " ".join(
            analyzed[
                analyzed["sentiment_label"]
                == "positive"
            ]["display_text"]
            .dropna()
            .astype(str)
        )

        if positive_text.strip():

            wc = WordCloud(
                width=800,
                height=450,
                background_color="white",
                colormap="Greens",
            ).generate(
                positive_text
            )

            st.image(
                wc.to_array(),
                use_container_width=True,
            )

        else:

            st.info(
                "No positive text available."
            )

    # ========================================================
    # COLLECTION HISTORY
    # ========================================================

    st.subheader("🕐 Collection History")

    history = load_collection_history()

    if not history.empty:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No collection history yet."
        )

    # ========================================================
    # CAMPAIGN INTELLIGENCE BRIEF
    # ========================================================

    st.subheader("📋 Intelligence Brief")

    st.caption(
        "This is a neutral monitoring summary of the "
        "online conversation captured by the app."
    )

    if st.button(
        "📝 Generate Intelligence Brief",
        use_container_width=True,
    ):

        brief = generate_intelligence_brief(
            analyzed,
            trends,
        )

        st.markdown(brief)

        st.download_button(
            "⬇️ Download Brief",
            brief,
            file_name="swedish_election_intelligence_brief.md",
            mime="text/markdown",
        )

    # ========================================================
    # RECENT CONTENT
    # ========================================================

    st.subheader("🗣️ Recent Conversation")

    recent = analyzed.sort_values(
        "collected_at",
        ascending=False,
    ).head(100)

    st.dataframe(
        recent[
            [
                "collected_at",
                "source",
                "sentiment_label",
                "sentiment_score",
                "party_mentioned",
                "leader_mentioned",
                "issue_mentioned",
                "engagement",
                "title",
                "display_text",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    st.subheader("💾 Export Data")

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        csv,
        file_name="swedish_election_intelligence.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# APPLICATION START
# ============================================================

def main():

    init_database()

    show_dashboard()


if __name__ == "__main__":
    main()
