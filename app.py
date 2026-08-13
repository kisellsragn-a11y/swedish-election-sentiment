import os
import sqlite3
import threading
from datetime import datetime, timezone
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
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

# IMPORTANT:
# We intentionally DO NOT use the old:
# cardiffnlp/twitter-xlm-roberta-base-sentiment
#
# That model was causing your sentencepiece.bpe.model error.
#
# This model is multilingual and uses DistilBERT.
SENTIMENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"


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
# API SECRETS
# ============================================================

def get_secret(name, default=""):
    """
    Get a Streamlit secret first, then environment variable.
    """
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    return os.environ.get(name, default)


REDDIT_CLIENT_ID = get_secret("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = get_secret("REDDIT_CLIENT_SECRET")
YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY")


# ============================================================
# SQLITE HELPERS
# ============================================================

_db_lock = threading.RLock()


def get_connection():
    """
    Create a SQLite connection configured to reduce
    'database is locked' errors.
    """
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
    )

    # Wait up to 30 seconds if another operation is writing.
    conn.execute("PRAGMA busy_timeout = 30000")

    # WAL allows readers while another process is writing.
    conn.execute("PRAGMA journal_mode = WAL")

    # Better durability without excessive locking.
    conn.execute("PRAGMA synchronous = NORMAL")

    return conn


def init_database():
    """
    Create database/tables safely.
    """
    with _db_lock:
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
                    issue_mentioned TEXT,
                    collected_at TEXT DEFAULT CURRENT_TIMESTAMP
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
                    issue_mentioned TEXT,
                    collected_at TEXT DEFAULT CURRENT_TIMESTAMP
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

            # Helpful indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reddit_sentiment
                ON reddit_posts(sentiment_label)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_youtube_sentiment
                ON youtube_comments(sentiment_label)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reddit_party
                ON reddit_posts(party_mentioned)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_youtube_party
                ON youtube_comments(party_mentioned)
            """)

            conn.commit()

        finally:
            conn.close()


# ============================================================
# TEXT HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""
    return str(value)


def detect_party(text):
    if not text:
        return None

    text_lower = safe_text(text).lower()

    party_keywords = {
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
        ],
        "Kristdemokraterna": [
            "kristdemokraterna",
            "kristdemokrat",
            "ebba busch",
        ],
        "Liberalerna": [
            "liberalerna",
            "liberal",
            "folkpartiet",
            "johan pehrson",
        ],
        "Centerpartiet": [
            "centerpartiet",
            "centerparti",
            "c-partiet",
            "muharrem demirok",
        ],
        "Miljöpartiet": [
            "miljöpartiet",
            "miljopartiet",
            "miljoparti",
            "miljöparti",
            "amanda lind",
        ],
        "Vänsterpartiet": [
            "vänsterpartiet",
            "vansterpartiet",
            "vänsterparti",
            "vansterparti",
            "nooshi dadgostar",
        ],
    }

    for party, keywords in party_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return party

    return None


def detect_issue(text):
    if not text:
        return None

    text_lower = safe_text(text).lower()

    issue_keywords = {
        "Immigration": [
            "invandring",
            "immigration",
            "migrant",
            "flykting",
            "asyl",
            "integration",
        ],
        "Crime": [
            "kriminalitet",
            "brott",
            "crime",
            "våld",
            "vald",
            "skjutning",
            "gäng",
            "gang",
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
            "bnp",
            "recession",
        ],
        "Climate": [
            "klimat",
            "climate",
            "miljö",
            "miljo",
            "koldioxid",
        ],
        "NATO": [
            "nato",
            "försvar",
            "forsvar",
            "defense",
            "militär",
            "militar",
        ],
        "Housing": [
            "bostad",
            "housing",
            "bostäder",
            "bostader",
            "hyra",
            "bostadsbrist",
        ],
        "Energy": [
            "elpris",
            "energi",
            "energy",
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
        ],
    }

    for issue, keywords in issue_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return issue

    return None


# ============================================================
# SENTIMENT MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """
    Load a multilingual DistilBERT model.

    This deliberately replaces the broken CardiffNLP
    XLM-RoBERTa model that was producing the
    sentencepiece.bpe.model parsing error.
    """
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        device=-1,
    )

    return classifier


def normalize_sentiment(result):
    """
    Convert model output into:
        positive
        negative
        neutral

    Returns:
        label, score
    """

    label = safe_text(result.get("label")).lower()
    score = float(result.get("score", 0.0))

    if "positive" in label:
        return "positive", abs(score)

    if "negative" in label:
        return "negative", -abs(score)

    return "neutral", 0.0


# ============================================================
# DATABASE INSERT HELPERS
# ============================================================

def insert_reddit_rows(rows):
    if not rows:
        return 0

    with _db_lock:
        conn = get_connection()

        try:
            conn.executemany(
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
                    issue_mentioned
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            conn.commit()
            return len(rows)

        finally:
            conn.close()


def insert_youtube_rows(rows):
    if not rows:
        return 0

    with _db_lock:
        conn = get_connection()

        try:
            conn.executemany(
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
                    issue_mentioned
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            conn.commit()
            return len(rows)

        finally:
            conn.close()


# ============================================================
# REDDIT COLLECTOR
# ============================================================

def collect_reddit(limit=150):
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return (
            0,
            "Reddit credentials are not configured. "
            "Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to Streamlit Secrets.",
        )

    try:
        import praw
    except ImportError:
        return (
            0,
            "PRAW is missing. Add praw to requirements.txt and redeploy.",
        )

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent="SwedishElectionMonitor/1.0",
        )

        # Test credentials.
        reddit.user.me

    except Exception as e:
        return 0, f"Reddit authentication failed: {e}"

    rows = []

    per_term = max(
        1,
        min(10, limit // max(1, len(SEARCH_TERMS))),
    )

    for subreddit_name in SUBREDDITS:
        for term in SEARCH_TERMS:

            try:
                subreddit = reddit.subreddit(subreddit_name)

                for post in subreddit.search(
                    term,
                    limit=per_term,
                    sort="new",
                ):
                    title = safe_text(post.title)
                    body = safe_text(post.selftext)
                    combined = f"{title} {body}"

                    party = detect_party(combined)
                    issue = detect_issue(combined)

                    rows.append(
                        (
                            safe_text(post.id),
                            subreddit_name,
                            safe_text(post.author),
                            title,
                            body,
                            int(post.score or 0),
                            int(post.num_comments or 0),
                            float(post.created_utc or 0),
                            safe_text(post.url),
                            safe_text(post.permalink),
                            party,
                            issue,
                        )
                    )

            except Exception:
                continue

    inserted = insert_reddit_rows(rows)

    return (
        inserted,
        f"Collected {inserted} Reddit posts.",
    )


# ============================================================
# YOUTUBE COLLECTOR
# ============================================================

def collect_youtube(max_results=20):
    if not YOUTUBE_API_KEY:
        return (
            0,
            "YouTube API key is not configured. "
            "Add YOUTUBE_API_KEY to Streamlit Secrets.",
        )

    try:
        from googleapiclient.discovery import build
    except ImportError:
        return (
            0,
            "Google API package is missing. "
            "Add google-api-python-client to requirements.txt.",
        )

    try:
        youtube = build(
            "youtube",
            "v3",
            developerKey=YOUTUBE_API_KEY,
        )
    except Exception as e:
        return 0, f"YouTube API initialization failed: {e}"

    rows = []

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

            for video in search_response.get("items", []):

                video_id = (
                    video.get("id", {})
                    .get("videoId")
                )

                if not video_id:
                    continue

                snippet = video.get("snippet", {})

                video_title = safe_text(
                    snippet.get("title")
                )

                try:
                    comments_response = (
                        youtube.commentThreads()
                        .list(
                            part="snippet",
                            videoId=video_id,
                            maxResults=50,
                            order="relevance",
                        )
                        .execute()
                    )

                    for item in comments_response.get(
                        "items",
                        [],
                    ):

                        comment_data = (
                            item
                            .get("snippet", {})
                            .get("topLevelComment", {})
                            .get("snippet", {})
                        )

                        text = safe_text(
                            comment_data.get(
                                "textDisplay"
                            )
                        )

                        comment_id = safe_text(
                            item.get("id")
                        )

                        party = detect_party(text)
                        issue = detect_issue(text)

                        rows.append(
                            (
                                comment_id,
                                video_id,
                                video_title,
                                safe_text(
                                    comment_data.get(
                                        "authorDisplayName"
                                    )
                                ),
                                text,
                                int(
                                    comment_data.get(
                                        "likeCount",
                                        0,
                                    )
                                    or 0
                                ),
                                safe_text(
                                    comment_data.get(
                                        "publishedAt"
                                    )
                                ),
                                party,
                                issue,
                            )
                        )

                except Exception:
                    continue

        except Exception:
            continue

    inserted = insert_youtube_rows(rows)

    return (
        inserted,
        f"Collected {inserted} YouTube comments.",
    )


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_database():
    """
    Analyze all currently unprocessed posts.

    IMPORTANT:
    We do NOT hold an SQLite connection open while the AI
    model is running. This is one of the fixes for
    'database is locked'.
    """

    # --------------------------------------------------------
    # Load model BEFORE opening SQLite.
    # --------------------------------------------------------

    try:
        classifier = load_sentiment_model()
    except Exception as e:
        return (
            0,
            f"Could not load sentiment model: {e}",
        )

    # --------------------------------------------------------
    # Read unprocessed data.
    # --------------------------------------------------------

    with _db_lock:
        conn = get_connection()

        try:
            reddit_rows = conn.execute(
                """
                SELECT id, title, text
                FROM reddit_posts
                WHERE sentiment_label IS NULL
                """
            ).fetchall()

            youtube_rows = conn.execute(
                """
                SELECT id, text
                FROM youtube_comments
                WHERE sentiment_label IS NULL
                """
            ).fetchall()

        finally:
            conn.close()

    analyzed_count = 0

    # --------------------------------------------------------
    # Analyze Reddit
    # --------------------------------------------------------

    reddit_updates = []

    for post_id, title, text in reddit_rows:

        combined = (
            f"{safe_text(title)} {safe_text(text)}"
        ).strip()

        if not combined:
            label = "neutral"
            score = 0.0

        else:
            try:
                result = classifier(
                    combined[:512],
                    truncation=True,
                    max_length=512,
                )[0]

                label, score = normalize_sentiment(
                    result
                )

            except Exception:
                label = "neutral"
                score = 0.0

        reddit_updates.append(
            (
                label,
                score,
                post_id,
            )
        )

    # --------------------------------------------------------
    # Analyze YouTube
    # --------------------------------------------------------

    youtube_updates = []

    for comment_id, text in youtube_rows:

        text = safe_text(text).strip()

        if not text:
            label = "neutral"
            score = 0.0

        else:
            try:
                result = classifier(
                    text[:512],
                    truncation=True,
                    max_length=512,
                )[0]

                label, score = normalize_sentiment(
                    result
                )

            except Exception:
                label = "neutral"
                score = 0.0

        youtube_updates.append(
            (
                label,
                score,
                comment_id,
            )
        )

    # --------------------------------------------------------
    # WRITE RESULTS IN SHORT TRANSACTIONS
    # --------------------------------------------------------

    with _db_lock:

        conn = get_connection()

        try:

            if reddit_updates:
                conn.executemany(
                    """
                    UPDATE reddit_posts
                    SET sentiment_label = ?,
                        sentiment_score = ?
                    WHERE id = ?
                    """,
                    reddit_updates,
                )

            if youtube_updates:
                conn.executemany(
                    """
                    UPDATE youtube_comments
                    SET sentiment_label = ?,
                        sentiment_score = ?
                    WHERE id = ?
                    """,
                    youtube_updates,
                )

            conn.commit()

            analyzed_count = (
                len(reddit_updates)
                + len(youtube_updates)
            )

        finally:
            conn.close()

    return (
        analyzed_count,
        "Sentiment analysis completed.",
    )


# ============================================================
# SUMMARY
# ============================================================

def generate_summary():
    with _db_lock:

        conn = get_connection()

        try:
            query = """
                SELECT
                    sentiment_label,
                    sentiment_score,
                    text,
                    party_mentioned,
                    issue_mentioned
                FROM reddit_posts
                WHERE sentiment_label IS NOT NULL

                UNION ALL

                SELECT
                    sentiment_label,
                    sentiment_score,
                    text,
                    party_mentioned,
                    issue_mentioned
                FROM youtube_comments
                WHERE sentiment_label IS NOT NULL
            """

            df = pd.read_sql_query(
                query,
                conn,
            )

        finally:
            conn.close()

    if df.empty:
        return None

    positive = (
        df["sentiment_label"]
        .eq("positive")
        .sum()
    )

    negative = (
        df["sentiment_label"]
        .eq("negative")
        .sum()
    )

    neutral = (
        df["sentiment_label"]
        .eq("neutral")
        .sum()
    )

    avg_sentiment = float(
        df["sentiment_score"]
        .fillna(0)
        .mean()
    )

    summary = {
        "date": datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d"),

        "total_posts": len(df),

        "positive_count": int(
            positive
        ),

        "negative_count": int(
            negative
        ),

        "neutral_count": int(
            neutral
        ),

        "avg_sentiment": avg_sentiment,
    }

    with _db_lock:

        conn = get_connection()

        try:

            conn.execute(
                """
                INSERT OR REPLACE INTO sentiment_summary
                (
                    date,
                    total_posts,
                    positive_count,
                    negative_count,
                    neutral_count,
                    avg_sentiment,
                    top_positive,
                    top_negative
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary["date"],
                    summary["total_posts"],
                    summary["positive_count"],
                    summary["negative_count"],
                    summary["neutral_count"],
                    summary["avg_sentiment"],
                    "N/A",
                    "N/A",
                ),
            )

            conn.commit()

        finally:
            conn.close()

    return summary


# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

def load_dashboard_data():

    with _db_lock:

        conn = get_connection()

        try:

            reddit_df = pd.read_sql_query(
                """
                SELECT *
                FROM reddit_posts
                WHERE sentiment_label IS NOT NULL
                """,
                conn,
            )

            youtube_df = pd.read_sql_query(
                """
                SELECT *
                FROM youtube_comments
                WHERE sentiment_label IS NOT NULL
                """,
                conn,
            )

            summary_df = pd.read_sql_query(
                """
                SELECT *
                FROM sentiment_summary
                ORDER BY date DESC
                LIMIT 30
                """,
                conn,
            )

        finally:
            conn.close()

    return (
        reddit_df,
        youtube_df,
        summary_df,
    )


# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard():

    st.title(
        "🇸🇪 Swedish Election Sentiment Monitor 2026"
    )

    election_date = datetime(
        2026,
        9,
        13,
    )

    now = datetime.now()

    days_until = max(
        0,
        (election_date - now).days,
    )

    st.markdown(
        f"""
        **Riksdagsval: September 13, 2026**
        | **{days_until} days until election**
        """
    )

    st.markdown(
        "AI-powered social media sentiment analysis "
        "for the Swedish general election."
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.header("🎛️ Controls")

        st.caption(
            f"AI model: `{SENTIMENT_MODEL}`"
        )

        # ----------------------------------------------------
        # COLLECT
        # ----------------------------------------------------

        if st.button(
            "🔄 Collect New Data",
            use_container_width=True,
        ):

            with st.spinner(
                "Collecting Reddit and YouTube data..."
            ):

                reddit_count, reddit_msg = (
                    collect_reddit()
                )

                youtube_count, youtube_msg = (
                    collect_youtube()
                )

            st.success(
                f"Reddit: {reddit_count}"
            )

            st.success(
                f"YouTube: {youtube_count}"
            )

            st.caption(reddit_msg)
            st.caption(youtube_msg)

        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        if st.button(
            "🧠 Analyze Sentiment",
            use_container_width=True,
        ):

            with st.spinner(
                "Loading AI model and analyzing posts..."
            ):

                analyzed, message = (
                    analyze_database()
                )

            if analyzed > 0:
                generate_summary()

                st.success(
                    f"Analyzed {analyzed} items."
                )

            else:
                st.warning(message)

        # ----------------------------------------------------
        # REFRESH
        # ----------------------------------------------------

        if st.button(
            "📊 Refresh Dashboard",
            use_container_width=True,
        ):
            st.rerun()

        st.divider()

        st.subheader("🇸🇪 Party Reference")

        for party, info in SWEDISH_PARTIES.items():

            st.markdown(
                f"**{info['abbrev']}** — {party}"
            )

    # ========================================================
    # LOAD DATA
    # ========================================================

    try:

        (
            reddit_df,
            youtube_df,
            summary_df,
        ) = load_dashboard_data()

    except sqlite3.OperationalError as e:

        st.error(
            f"Database error: {e}"
        )

        st.info(
            "If the database is temporarily busy, "
            "wait a few seconds and click Refresh Dashboard."
        )

        return

    # ========================================================
    # PREPARE DATA
    # ========================================================

    if not reddit_df.empty:

        reddit_df["source"] = "Reddit"

        reddit_df["text"] = (
            reddit_df["title"]
            .fillna("")
            .astype(str)
            + " "
            + reddit_df["text"]
            .fillna("")
            .astype(str)
        )

    else:

        reddit_df = pd.DataFrame(
            columns=[
                "source",
                "text",
                "sentiment_label",
                "sentiment_score",
                "collected_at",
                "party_mentioned",
                "issue_mentioned",
            ]
        )

    if not youtube_df.empty:

        youtube_df["source"] = "YouTube"

    else:

        youtube_df = pd.DataFrame(
            columns=[
                "source",
                "text",
                "sentiment_label",
                "sentiment_score",
                "collected_at",
                "party_mentioned",
                "issue_mentioned",
            ]
        )

    all_data = pd.concat(
        [
            reddit_df[
                [
                    "source",
                    "text",
                    "sentiment_label",
                    "sentiment_score",
                    "collected_at",
                    "party_mentioned",
                    "issue_mentioned",
                ]
            ],
            youtube_df[
                [
                    "source",
                    "text",
                    "sentiment_label",
                    "sentiment_score",
                    "collected_at",
                    "party_mentioned",
                    "issue_mentioned",
                ]
            ],
        ],
        ignore_index=True,
    )

    # ========================================================
    # EMPTY STATE
    # ========================================================

    if all_data.empty:

        st.info(
            """
            No analyzed data yet.

            1. Add your API keys to Streamlit Secrets.
            2. Click **Collect New Data**.
            3. Click **Analyze Sentiment**.
            """
        )

        return

    # ========================================================
    # TOP METRICS
    # ========================================================

    total = len(all_data)

    positive_count = (
        all_data["sentiment_label"]
        .eq("positive")
        .sum()
    )

    negative_count = (
        all_data["sentiment_label"]
        .eq("negative")
        .sum()
    )

    positive_pct = (
        positive_count / total * 100
        if total
        else 0
    )

    negative_pct = (
        negative_count / total * 100
        if total
        else 0
    )

    avg_score = (
        all_data["sentiment_score"]
        .fillna(0)
        .mean()
    )

    party_mentions = (
        all_data["party_mentioned"]
        .notna()
        .sum()
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:
        st.metric(
            "Total Posts",
            total,
        )

    with col2:
        st.metric(
            "Positive %",
            f"{positive_pct:.1f}%",
        )

    with col3:
        st.metric(
            "Negative %",
            f"{negative_pct:.1f}%",
        )

    with col4:
        st.metric(
            "Avg Sentiment",
            f"{avg_score:.3f}",
        )

    with col5:
        st.metric(
            "Party Mentions",
            int(party_mentions),
        )

    # ========================================================
    # SENTIMENT DISTRIBUTION
    # ========================================================

    col_left, col_right = st.columns(2)

    with col_left:

        st.subheader(
            "Sentiment Distribution"
        )

        sentiment_counts = (
            all_data["sentiment_label"]
            .value_counts()
        )

        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Overall sentiment",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col_right:

        st.subheader(
            "Sentiment by Source"
        )

        source_sentiment = (
            all_data
            .groupby(
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
            barmode="group",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # PARTY + ISSUES
    # ========================================================

    col_left, col_right = st.columns(2)

    with col_left:

        st.subheader(
            "Sentiment by Party"
        )

        party_data = all_data[
            all_data["party_mentioned"]
            .notna()
        ]

        if not party_data.empty:

            party_sentiment = (
                party_data
                .groupby(
                    [
                        "party_mentioned",
                        "sentiment_label",
                    ]
                )
                .size()
                .reset_index(
                    name="count"
                )
            )

            fig = px.bar(
                party_sentiment,
                x="party_mentioned",
                y="count",
                color="sentiment_label",
                barmode="group",
            )

            fig.update_layout(
                xaxis_title="Party",
                yaxis_title="Mentions",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No party mentions found yet."
            )

    with col_right:

        st.subheader(
            "Top Issues Discussed"
        )

        issue_data = all_data[
            all_data["issue_mentioned"]
            .notna()
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
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No issues detected yet."
            )

    # ========================================================
    # SENTIMENT TREND
    # ========================================================

    st.subheader(
        "📈 Sentiment Trend Over Time"
    )

    if not summary_df.empty:

        trend_df = (
            summary_df
            .sort_values("date")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trend_df["date"],
                y=trend_df["avg_sentiment"],
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
            yaxis_title="Sentiment Score",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.info(
            "Trend data will appear after sentiment analysis."
        )

    # ========================================================
    # WORD CLOUDS
    # ========================================================

    st.subheader(
        "☁️ Sentiment Word Clouds"
    )

    try:

        from wordcloud import WordCloud

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### Negative Posts"
            )

            negative_text = " ".join(
                all_data[
                    all_data[
                        "sentiment_label"
                    ]
                    == "negative"
                ]["text"]
                .dropna()
                .astype(str)
            )

            if negative_text.strip():

                wordcloud = WordCloud(
                    width=800,
                    height=500,
                    background_color="white",
                ).generate(
                    negative_text
                )

                st.image(
                    wordcloud.to_array(),
                    use_container_width=True,
                )

            else:

                st.info(
                    "No negative posts yet."
                )

        with col2:

            st.markdown(
                "### Positive Posts"
            )

            positive_text = " ".join(
                all_data[
                    all_data[
                        "sentiment_label"
                    ]
                    == "positive"
                ]["text"]
                .dropna()
                .astype(str)
            )

            if positive_text.strip():

                wordcloud = WordCloud(
                    width=800,
                    height=500,
                    background_color="white",
                ).generate(
                    positive_text
                )

                st.image(
                    wordcloud.to_array(),
                    use_container_width=True,
                )

            else:

                st.info(
                    "No positive posts yet."
                )

    except ImportError:

        st.info(
            "WordCloud is not installed. "
            "Add wordcloud to requirements.txt."
        )

    # ========================================================
    # RECENT POSTS
    # ========================================================

    st.subheader(
        "📰 Recent Posts"
    )

    display_df = all_data[
        [
            "collected_at",
            "source",
            "sentiment_label",
            "sentiment_score",
            "party_mentioned",
            "issue_mentioned",
            "text",
        ]
    ].copy()

    display_df = (
        display_df
        .sort_values(
            "collected_at",
            ascending=False,
        )
        .head(50)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader(
        "⬇️ Download Data"
    )

    csv = all_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "swedish_election_sentiment.csv",
        "text/csv",
        use_container_width=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:
        init_database()

    except Exception as e:

        st.error(
            f"Could not initialize database: {e}"
        )

        st.stop()

    show_dashboard()


if __name__ == "__main__":
    main()
