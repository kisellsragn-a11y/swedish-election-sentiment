import os
import sqlite3
import time
from datetime import datetime
from collections import Counter

import pandas as pd
import requests
import praw

from googleapiclient.discovery import build
from transformers import pipeline

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Swedish Election Sentiment 2026",
    page_icon="🇸🇪",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

SWEDISH_PARTIES = {
    "Socialdemokraterna": {
        "leader": "Magdalena Andersson",
        "abbrev": "S",
        "bloc": "left"
    },
    "Moderaterna": {
        "leader": "Ulf Kristersson",
        "abbrev": "M",
        "bloc": "right"
    },
    "Sverigedemokraterna": {
        "leader": "Jimmie Åkesson",
        "abbrev": "SD",
        "bloc": "right"
    },
    "Kristdemokraterna": {
        "leader": "Ebba Busch",
        "abbrev": "KD",
        "bloc": "right"
    },
    "Liberalerna": {
        "leader": "Johan Pehrson",
        "abbrev": "L",
        "bloc": "right"
    },
    "Centerpartiet": {
        "leader": "Muharrem Demirok",
        "abbrev": "C",
        "bloc": "center"
    },
    "Miljöpartiet": {
        "leader": "Amanda Lind",
        "abbrev": "MP",
        "bloc": "left"
    },
    "Vänsterpartiet": {
        "leader": "Nooshi Dadgostar",
        "abbrev": "V",
        "bloc": "left"
    },
}


# ============================================================
# API SECRETS
# ============================================================

try:
    REDDIT_CLIENT_ID = st.secrets.get(
        "REDDIT_CLIENT_ID",
        os.environ.get("REDDIT_CLIENT_ID", "")
    )

    REDDIT_CLIENT_SECRET = st.secrets.get(
        "REDDIT_CLIENT_SECRET",
        os.environ.get("REDDIT_CLIENT_SECRET", "")
    )

    YOUTUBE_API_KEY = st.secrets.get(
        "YOUTUBE_API_KEY",
        os.environ.get("YOUTUBE_API_KEY", "")
    )

except Exception:
    REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
    YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")


# ============================================================
# DATABASE
# ============================================================

DB_PATH = "swedish_election_2026.db"


def get_connection():
    """
    Create a SQLite connection designed for Streamlit.

    WAL + busy_timeout greatly reduces 'database is locked'
    errors when Streamlit reruns the app.
    """

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False
    )

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    return conn


def execute_with_retry(operation, retries=5, delay=0.5):
    """
    Retry SQLite operations if another operation temporarily
    has the database locked.
    """

    last_error = None

    for attempt in range(retries):
        try:
            return operation()

        except sqlite3.OperationalError as e:
            last_error = e

            if "locked" not in str(e).lower():
                raise

            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    raise last_error


def init_database():

    def _init():

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

        conn.commit()
        conn.close()

    execute_with_retry(_init)


# ============================================================
# SEARCH TERMS
# ============================================================

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
    "politics"
]


YOUTUBE_QUERIES = [
    "riksdagsval 2026",
    "svensk politik 2026",
    "valdebatt 2026",
    "Magdalena Andersson",
    "Ulf Kristersson",
    "Jimmie Åkesson",
    "Sverigedemokraterna"
]


# ============================================================
# SWEDISH STOP WORDS
# ============================================================

SWEDISH_STOPWORDS = {
    "och", "att", "det", "som", "för", "är", "en", "ett",
    "den", "de", "av", "på", "i", "till", "med", "om",
    "har", "jag", "du", "han", "hon", "vi", "ni", "man",
    "inte", "men", "så", "kan", "ska", "var", "vara",
    "blir", "blev", "där", "här", "från", "eller", "efter",
    "innan", "också", "bara", "alla", "något", "någon",
    "några", "mycket", "mer", "mest", "sin", "sitt", "sina",
    "denna", "detta", "dessa", "ett", "en", "mig", "dig",
    "sig", "oss", "er", "dom", "då", "nu", "ut", "upp",
    "ned", "ner", "över", "under", "igen", "väl", "ju",
    "så", "vad", "hur", "varför", "vem", "vilken", "vilket",
    "vilka", "än", "också", "får", "få", "gör", "gjort",
    "göra", "går", "gick", "kom", "kommer", "kommer",
    "säger", "sa", "hade", "haft", "skulle", "kunna",
    "måste", "bör", "blir", "blivit", "är", "varit",
    "the", "and", "that", "this", "with", "from", "for",
    "you", "your", "they", "them", "have", "has", "not"
}


def clean_wordcloud_text(text):

    if not text:
        return ""

    words = str(text).lower().split()

    cleaned = []

    for word in words:

        word = word.strip(
            ".,!?;:\"'()[]{}<>|/\\+-_=*#@"
        )

        if not word:
            continue

        if len(word) < 3:
            continue

        if word in SWEDISH_STOPWORDS:
            continue

        if word.startswith("http"):
            continue

        if "www." in word:
            continue

        cleaned.append(word)

    return " ".join(cleaned)


# ============================================================
# PARTY DETECTION
# ============================================================

PARTY_KEYWORDS = {

    "Socialdemokraterna": [
        "socialdemokraterna",
        "socialdemokrat",
        "sosse",
        "magdalena andersson",
        "s-partiet",
        " s "
    ],

    "Moderaterna": [
        "moderaterna",
        "moderat",
        "ulf kristersson",
        "m-partiet",
        " moderaterna "
    ],

    "Sverigedemokraterna": [
        "sverigedemokraterna",
        "sverigedemokrat",
        "sd",
        "jimmie åkesson",
        "jimmie akesson"
    ],

    "Kristdemokraterna": [
        "kristdemokraterna",
        "kristdemokrat",
        "kd",
        "ebba busch"
    ],

    "Liberalerna": [
        "liberalerna",
        "liberal",
        "folkpartiet",
        "johan pehrson",
        "fp"
    ],

    "Centerpartiet": [
        "centerpartiet",
        "centerparti",
        "c-partiet",
        "muharrem demirok"
    ],

    "Miljöpartiet": [
        "miljöpartiet",
        "miljopartiet",
        "miljoparti",
        "miljöparti",
        "mp",
        "amanda lind"
    ],

    "Vänsterpartiet": [
        "vänsterpartiet",
        "vansterpartiet",
        "vänsterparti",
        "vansterparti",
        "nooshi dadgostar",
        "v-partiet"
    ],
}


def detect_parties(text):

    if not text:
        return []

    text_lower = " " + str(text).lower() + " "

    found = []

    for party, keywords in PARTY_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text_lower:
                found.append(party)
                break

    return found


def detect_party(text):

    parties = detect_parties(text)

    if not parties:
        return None

    return ", ".join(parties)


# ============================================================
# ISSUE DETECTION
# ============================================================

ISSUE_KEYWORDS = {

    "Immigration": [
        "invandring",
        "immigration",
        "migrant",
        "flykting",
        "asyl",
        "integration"
    ],

    "Crime": [
        "kriminalitet",
        "brott",
        "crime",
        "våld",
        "vald",
        "skjutning",
        "gäng",
        "gang"
    ],

    "Healthcare": [
        "sjukvård",
        "sjukvard",
        "vård",
        "vard",
        "healthcare",
        "sjukhus",
        "läkare",
        "lakare"
    ],

    "Education": [
        "skola",
        "utbildning",
        "school",
        "lärare",
        "larare",
        "elever"
    ],

    "Economy": [
        "ekonomi",
        "economy",
        "inflation",
        "priser",
        "bnp",
        "recession"
    ],

    "Climate": [
        "klimat",
        "climate",
        "miljö",
        "miljo",
        "koldioxid"
    ],

    "NATO": [
        "nato",
        "försvar",
        "forsvar",
        "defense",
        "militär",
        "militar"
    ],

    "Housing": [
        "bostad",
        "housing",
        "bostäder",
        "bostader",
        "hyra",
        "bostadsbrist"
    ],

    "Energy": [
        "elpris",
        "energi",
        "energy",
        "el",
        "kärnkraft",
        "karnkraft",
        "vindkraft"
    ],

    "Welfare": [
        "bidrag",
        "welfare",
        "försörjningsstöd",
        "forsorjningsstod",
        "socialbidrag",
        "pension"
    ]
}


def detect_issues(text):

    if not text:
        return []

    text_lower = str(text).lower()

    found = []

    for issue, keywords in ISSUE_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text_lower:
                found.append(issue)
                break

    return found


def detect_issue(text):

    issues = detect_issues(text)

    if not issues:
        return None

    return ", ".join(issues)


# ============================================================
# REDDIT COLLECTION
# ============================================================

def collect_reddit(limit=300):

    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:

        return 0, 0, (
            "ERROR: Reddit API credentials are not configured. "
            "Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to Streamlit Secrets."
        )

    try:

        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent="SwedishElectionMonitor/1.0"
        )

    except Exception as e:

        return 0, 0, f"ERROR: Reddit authentication failed: {e}"


    def _collect():

        conn = get_connection()
        cursor = conn.cursor()

        new_count = 0
        existing_count = 0

        limit_per_term = max(
            1,
            limit // max(1, len(SEARCH_TERMS))
        )

        for subreddit_name in SUBREDDITS:

            try:
                subreddit = reddit.subreddit(subreddit_name)

            except Exception:
                continue

            for term in SEARCH_TERMS:

                try:

                    posts = subreddit.search(
                        term,
                        limit=limit_per_term,
                        sort="new"
                    )

                    for post in posts:

                        title = post.title or ""
                        body = post.selftext or ""

                        full_text = f"{title} {body}"

                        parties = detect_party(full_text)
                        issues = detect_issue(full_text)

                        cursor.execute(
                            "SELECT 1 FROM reddit_posts WHERE id = ?",
                            (post.id,)
                        )

                        exists = cursor.fetchone()

                        if exists:

                            existing_count += 1
                            continue

                        cursor.execute(
                            """
                            INSERT INTO reddit_posts
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
                            (
                                post.id,
                                subreddit_name,
                                str(post.author) if post.author else "",
                                title,
                                body,
                                post.score,
                                post.num_comments,
                                post.created_utc,
                                post.url,
                                post.permalink,
                                parties,
                                issues
                            )
                        )

                        new_count += 1

                except Exception:
                    continue

        conn.commit()
        conn.close()

        return new_count, existing_count

    try:

        new_count, existing_count = execute_with_retry(_collect)

        return (
            new_count,
            existing_count,
            f"Reddit: {new_count} new posts, "
            f"{existing_count} already stored."
        )

    except Exception as e:

        return 0, 0, f"ERROR collecting Reddit: {e}"


# ============================================================
# YOUTUBE COLLECTION
# ============================================================

def collect_youtube(max_results=30):

    if not YOUTUBE_API_KEY:

        return 0, 0, (
            "ERROR: YouTube API key is not configured. "
            "Add YOUTUBE_API_KEY to Streamlit Secrets."
        )

    try:

        youtube = build(
            "youtube",
            "v3",
            developerKey=YOUTUBE_API_KEY
        )

    except Exception as e:

        return 0, 0, f"ERROR: YouTube API initialization failed: {e}"


    def _collect():

        conn = get_connection()
        cursor = conn.cursor()

        new_count = 0
        existing_count = 0

        for query in YOUTUBE_QUERIES:

            try:

                search_response = (
                    youtube.search()
                    .list(
                        q=query,
                        part="id,snippet",
                        maxResults=max_results,
                        type="video",
                        order="relevance"
                    )
                    .execute()
                )

            except Exception:
                continue


            for video in search_response.get("items", []):

                try:

                    video_id = video["id"]["videoId"]
                    video_title = video["snippet"]["title"]

                except Exception:
                    continue


                try:

                    comments_response = (
                        youtube.commentThreads()
                        .list(
                            part="snippet",
                            videoId=video_id,
                            maxResults=50,
                            order="relevance"
                        )
                        .execute()
                    )

                except Exception:
                    continue


                for item in comments_response.get("items", []):

                    try:

                        comment = (
                            item["snippet"]
                            ["topLevelComment"]
                            ["snippet"]
                        )

                        comment_id = item["id"]

                        cursor.execute(
                            "SELECT 1 FROM youtube_comments WHERE id = ?",
                            (comment_id,)
                        )

                        exists = cursor.fetchone()

                        if exists:

                            existing_count += 1
                            continue


                        text = comment.get("textDisplay", "")

                        parties = detect_party(text)
                        issues = detect_issue(text)

                        cursor.execute(
                            """
                            INSERT INTO youtube_comments
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
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                comment_id,
                                video_id,
                                video_title,
                                comment.get(
                                    "authorDisplayName",
                                    ""
                                ),
                                text,
                                comment.get(
                                    "likeCount",
                                    0
                                ),
                                comment.get(
                                    "publishedAt",
                                    ""
                                ),
                                parties,
                                issues
                            )
                        )

                        new_count += 1

                    except Exception:
                        continue


        conn.commit()
        conn.close()

        return new_count, existing_count


    try:

        new_count, existing_count = execute_with_retry(_collect)

        return (
            new_count,
            existing_count,
            f"YouTube: {new_count} new comments, "
            f"{existing_count} already stored."
        )

    except Exception as e:

        return 0, 0, f"ERROR collecting YouTube: {e}"


# ============================================================
# SENTIMENT MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )


def normalize_sentiment(result, classifier):

    label = str(result["label"]).lower()
    score = float(result["score"])

    # Handle models that expose LABEL_0 etc.
    try:

        id2label = classifier.model.config.id2label

        label_id = None

        if label.startswith("label_"):
            label_id = int(label.split("_")[-1])

        if label_id is not None and label_id in id2label:

            label = str(
                id2label[label_id]
            ).lower()

    except Exception:
        pass


    if "positive" in label:

        return "positive", score

    if "negative" in label:

        return "negative", -score

    return "neutral", 0.0


# ============================================================
# ANALYZE DATABASE
# ============================================================

def analyze_database():

    try:

        classifier = load_sentiment_model()

    except Exception as e:

        return 0, f"Could not load sentiment model: {e}"


    def _analyze():

        conn = get_connection()
        cursor = conn.cursor()

        analyzed_count = 0


        # ----------------------------------------------------
        # REDDIT
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id, title, text
            FROM reddit_posts
            WHERE sentiment_label IS NULL
            """
        )

        reddit_posts = cursor.fetchall()


        for post_id, title, text in reddit_posts:

            full_text = f"{title or ''} {text or ''}".strip()

            if not full_text:

                label = "neutral"
                score = 0.0

            else:

                try:

                    result = classifier(
                        full_text[:512]
                    )[0]

                    label, score = normalize_sentiment(
                        result,
                        classifier
                    )

                except Exception:

                    label = "neutral"
                    score = 0.0


            cursor.execute(
                """
                UPDATE reddit_posts
                SET sentiment_label = ?,
                    sentiment_score = ?
                WHERE id = ?
                """,
                (
                    label,
                    score,
                    post_id
                )
            )

            analyzed_count += 1


        # ----------------------------------------------------
        # YOUTUBE
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id, text
            FROM youtube_comments
            WHERE sentiment_label IS NULL
            """
        )

        youtube_comments = cursor.fetchall()


        for comment_id, text in youtube_comments:

            text = text or ""

            if not text.strip():

                label = "neutral"
                score = 0.0

            else:

                try:

                    result = classifier(
                        text[:512]
                    )[0]

                    label, score = normalize_sentiment(
                        result,
                        classifier
                    )

                except Exception:

                    label = "neutral"
                    score = 0.0


            cursor.execute(
                """
                UPDATE youtube_comments
                SET sentiment_label = ?,
                    sentiment_score = ?
                WHERE id = ?
                """,
                (
                    label,
                    score,
                    comment_id
                )
            )

            analyzed_count += 1


        conn.commit()
        conn.close()

        return analyzed_count


    try:

        count = execute_with_retry(_analyze)

        return (
            count,
            f"Successfully analyzed {count} new items."
        )

    except Exception as e:

        return 0, f"ERROR during sentiment analysis: {e}"


# ============================================================
# SUMMARY
# ============================================================

def generate_summary():

    def _generate():

        conn = get_connection()

        query = """
            SELECT
                sentiment_label,
                sentiment_score,
                text,
                party_mentioned,
                issue_mentioned,
                'Reddit' AS source
            FROM reddit_posts
            WHERE sentiment_label IS NOT NULL

            UNION ALL

            SELECT
                sentiment_label,
                sentiment_score,
                text,
                party_mentioned,
                issue_mentioned,
                'YouTube' AS source
            FROM youtube_comments
            WHERE sentiment_label IS NOT NULL
        """

        df = pd.read_sql_query(
            query,
            conn
        )

        conn.close()

        if len(df) == 0:
            return None


        summary = {

            "date":
                datetime.now().strftime("%Y-%m-%d"),

            "total_posts":
                len(df),

            "positive_count":
                len(
                    df[
                        df["sentiment_label"]
                        == "positive"
                    ]
                ),

            "negative_count":
                len(
                    df[
                        df["sentiment_label"]
                        == "negative"
                    ]
                ),

            "neutral_count":
                len(
                    df[
                        df["sentiment_label"]
                        == "neutral"
                    ]
                ),

            "avg_sentiment":
                df["sentiment_score"].mean()
        }


        conn = get_connection()
        cursor = conn.cursor()


        cursor.execute(
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
                "N/A"
            )
        )


        conn.commit()
        conn.close()

        return summary


    return execute_with_retry(_generate)


# ============================================================
# DASHBOARD DATA
# ============================================================

def load_dashboard_data():

    def _load():

        conn = get_connection()

        reddit_df = pd.read_sql_query(
            """
            SELECT *
            FROM reddit_posts
            WHERE sentiment_label IS NOT NULL
            """,
            conn
        )

        youtube_df = pd.read_sql_query(
            """
            SELECT *
            FROM youtube_comments
            WHERE sentiment_label IS NOT NULL
            """,
            conn
        )

        summary_df = pd.read_sql_query(
            """
            SELECT *
            FROM sentiment_summary
            ORDER BY date DESC
            LIMIT 30
            """,
            conn
        )

        conn.close()

        return (
            reddit_df,
            youtube_df,
            summary_df
        )

    return execute_with_retry(_load)


# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard():

    st.title(
        "🇸🇪 Swedish Election Sentiment Monitor 2026"
    )


    election_date = datetime(2026, 9, 13)

    days_until = (
        election_date - datetime.now()
    ).days

    st.markdown(
        f"**Riksdagsval: September 13, 2026** "
        f"| {days_until} days until election"
    )

    st.markdown(
        "AI-powered social media sentiment analysis "
        "for the Swedish general election."
    )


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.title("🎛️ Controls")

        st.caption(
            "Manual mode — data is collected only when "
            "you press the button."
        )


        # ----------------------------------------------------
        # COLLECT
        # ----------------------------------------------------

        if st.button(
            "🔄 Collect New Data",
            use_container_width=True
        ):

            with st.spinner(
                "Collecting new Reddit and YouTube data..."
            ):

                reddit_new, reddit_existing, reddit_msg = (
                    collect_reddit()
                )

                youtube_new, youtube_existing, youtube_msg = (
                    collect_youtube()
                )


            st.success(
                "Collection complete."
            )

            st.write(reddit_msg)
            st.write(youtube_msg)

            st.info(
                f"🆕 Total new items: "
                f"{reddit_new + youtube_new}"
            )


        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        if st.button(
            "🧠 Analyze Sentiment",
            use_container_width=True
        ):

            with st.spinner(
                "Running XLM-R sentiment analysis..."
            ):

                analyzed, message = (
                    analyze_database()
                )

                if analyzed > 0:

                    try:
                        generate_summary()
                    except Exception:
                        pass


            if analyzed > 0:

                st.success(
                    f"Analyzed {analyzed} new items."
                )

            else:

                if message.startswith("ERROR"):

                    st.error(message)

                else:

                    st.info(
                        "No unanalyzed items found."
                    )


        # ----------------------------------------------------
        # REFRESH
        # ----------------------------------------------------

        if st.button(
            "📊 Refresh Dashboard",
            use_container_width=True
        ):

            st.rerun()


        st.markdown("---")


        # ----------------------------------------------------
        # DATABASE STATUS
        # ----------------------------------------------------

        st.subheader("📦 Database Status")

        try:

            conn = get_connection()

            reddit_total = conn.execute(
                "SELECT COUNT(*) FROM reddit_posts"
            ).fetchone()[0]

            youtube_total = conn.execute(
                "SELECT COUNT(*) FROM youtube_comments"
            ).fetchone()[0]

            reddit_unanalyzed = conn.execute(
                """
                SELECT COUNT(*)
                FROM reddit_posts
                WHERE sentiment_label IS NULL
                """
            ).fetchone()[0]

            youtube_unanalyzed = conn.execute(
                """
                SELECT COUNT(*)
                FROM youtube_comments
                WHERE sentiment_label IS NULL
                """
            ).fetchone()[0]

            conn.close()


            st.metric(
                "Reddit items",
                reddit_total
            )

            st.metric(
                "YouTube comments",
                youtube_total
            )

            st.caption(
                f"Unanalyzed: "
                f"{reddit_unanalyzed + youtube_unanalyzed}"
            )

        except Exception as e:

            st.warning(
                f"Database status unavailable: {e}"
            )


        st.markdown("---")


        # ----------------------------------------------------
        # PARTY REFERENCE
        # ----------------------------------------------------

        st.subheader("Party Reference")

        for party, info in SWEDISH_PARTIES.items():

            st.markdown(
                f"**{info['abbrev']}** - {party}"
            )


    # ========================================================
    # LOAD DATA
    # ========================================================

    try:

        (
            reddit_df,
            youtube_df,
            summary_df
        ) = load_dashboard_data()

    except Exception as e:

        st.error(
            f"Could not read database: {e}"
        )

        return


    # ========================================================
    # PREPARE DATA
    # ========================================================

    reddit_df["source"] = "Reddit"
    youtube_df["source"] = "YouTube"


    if len(reddit_df) > 0:

        reddit_df["text"] = (
            reddit_df["title"].fillna("")
            + " "
            + reddit_df["text"].fillna("")
        )


    required_columns = [
        "source",
        "text",
        "sentiment_label",
        "sentiment_score",
        "collected_at",
        "party_mentioned",
        "issue_mentioned"
    ]


    all_data = pd.concat(
        [
            reddit_df[required_columns],
            youtube_df[required_columns]
        ],
        ignore_index=True
    )


    # ========================================================
    # EMPTY STATE
    # ========================================================

    if len(all_data) == 0:

        st.info(
            "No analyzed data yet. "
            "Press 'Collect New Data', then "
            "'Analyze Sentiment'."
        )

        return


    # ========================================================
    # TOP METRICS
    # ========================================================

    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "Total Posts",
            len(all_data)
        )


    with col2:

        positive_count = len(
            all_data[
                all_data["sentiment_label"]
                == "positive"
            ]
        )

        pos_pct = (
            positive_count
            / len(all_data)
            * 100
        )

        st.metric(
            "Positive %",
            f"{pos_pct:.1f}%"
        )


    with col3:

        negative_count = len(
            all_data[
                all_data["sentiment_label"]
                == "negative"
            ]
        )

        neg_pct = (
            negative_count
            / len(all_data)
            * 100
        )

        st.metric(
            "Negative %",
            f"{neg_pct:.1f}%"
        )


    with col4:

        avg_score = (
            all_data["sentiment_score"]
            .mean()
        )

        st.metric(
            "Avg Sentiment",
            f"{avg_score:.3f}"
        )


    with col5:

        party_mentions = (
            all_data["party_mentioned"]
            .notna()
            .sum()
        )

        st.metric(
            "Political Mentions",
            party_mentions
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
            color=sentiment_counts.index,
            color_discrete_map={
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # SENTIMENT BY SOURCE
    # ========================================================

    with col_right:

        st.subheader(
            "Sentiment by Source"
        )

        source_sentiment = (
            all_data
            .groupby(
                [
                    "source",
                    "sentiment_label"
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
            color_discrete_map={
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # PARTY ANALYSIS
    # ========================================================

    col_left2, col_right2 = st.columns(2)


    with col_left2:

        st.subheader(
            "Sentiment by Party"
        )

        party_data = all_data[
            all_data["party_mentioned"]
            .notna()
        ]


        if len(party_data) > 0:

            party_rows = []

            for _, row in party_data.iterrows():

                parties = str(
                    row["party_mentioned"]
                ).split(",")

                for party in parties:

                    party = party.strip()

                    if party:

                        party_rows.append({
                            "party": party,
                            "sentiment_label":
                                row["sentiment_label"]
                        })


            if party_rows:

                party_expanded = pd.DataFrame(
                    party_rows
                )

                party_sentiment = (
                    party_expanded
                    .groupby(
                        [
                            "party",
                            "sentiment_label"
                        ]
                    )
                    .size()
                    .reset_index(
                        name="count"
                    )
                )


                fig = px.bar(
                    party_sentiment,
                    x="party",
                    y="count",
                    color="sentiment_label",
                    color_discrete_map={
                        "positive": "#2ecc71",
                        "negative": "#e74c3c",
                        "neutral": "#95a5a6"
                    }
                )

                fig.update_layout(
                    xaxis_title="Party",
                    yaxis_title="Mentions"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "No party mentions found yet."
                )

        else:

            st.info(
                "No party mentions found yet."
            )


    # ========================================================
    # TOP ISSUES
    # ========================================================

    with col_right2:

        st.subheader(
            "Top Issues Discussed"
        )

        issue_data = all_data[
            all_data["issue_mentioned"]
            .notna()
        ]


        if len(issue_data) > 0:

            issue_rows = []

            for _, row in issue_data.iterrows():

                issues = str(
                    row["issue_mentioned"]
                ).split(",")

                for issue in issues:

                    issue = issue.strip()

                    if issue:

                        issue_rows.append(issue)


            if issue_rows:

                issue_counts = (
                    pd.Series(issue_rows)
                    .value_counts()
                    .head(10)
                )


                fig = px.bar(
                    x=issue_counts.index,
                    y=issue_counts.values,
                    labels={
                        "x": "Issue",
                        "y": "Mentions"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            st.info(
                "No issue mentions found yet."
            )


    # ========================================================
    # SENTIMENT TREND
    # ========================================================

    st.subheader(
        "Sentiment Trend Over Time"
    )


    if len(summary_df) > 0:

        summary_plot = (
            summary_df
            .sort_values("date")
        )


        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=summary_plot["date"],
                y=summary_plot["avg_sentiment"],
                mode="lines+markers",
                name="Avg Sentiment",
                line=dict(
                    color="#3498db",
                    width=3
                )
            )
        )


        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray"
        )


        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment Score"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # WORD CLOUDS
    # ========================================================

    col_wc1, col_wc2 = st.columns(2)


    with col_wc1:

        st.subheader(
            "☁️ Word Cloud — Negative Posts"
        )


        negative_text = " ".join(
            all_data[
                all_data["sentiment_label"]
                == "negative"
            ]["text"]
            .dropna()
            .astype(str)
            .map(clean_wordcloud_text)
        )


        if negative_text.strip():

            wordcloud = WordCloud(
                width=800,
                height=500,
                background_color="white",
                colormap="Reds",
                max_words=150,
                collocations=False
            ).generate(
                negative_text
            )


            st.image(
                wordcloud.to_array(),
                use_container_width=True
            )

        else:

            st.info(
                "No negative text available."
            )


    with col_wc2:

        st.subheader(
            "☁️ Word Cloud — Positive Posts"
        )


        positive_text = " ".join(
            all_data[
                all_data["sentiment_label"]
                == "positive"
            ]["text"]
            .dropna()
            .astype(str)
            .map(clean_wordcloud_text)
        )


        if positive_text.strip():

            wordcloud = WordCloud(
                width=800,
                height=500,
                background_color="white",
                colormap="Greens",
                max_words=150,
                collocations=False
            ).generate(
                positive_text
            )


            st.image(
                wordcloud.to_array(),
                use_container_width=True
            )

        else:

            st.info(
                "No positive text available."
            )


    # ========================================================
    # MOST MENTIONED PARTIES
    # ========================================================

    st.subheader(
        "👥 Most Mentioned Parties"
    )


    party_counter = Counter()


    for value in all_data[
        "party_mentioned"
    ].dropna():

        for party in str(value).split(","):

            party = party.strip()

            if party:
                party_counter[party] += 1


    if party_counter:

        party_df = pd.DataFrame(
            party_counter.items(),
            columns=[
                "Party",
                "Mentions"
            ]
        ).sort_values(
            "Mentions",
            ascending=False
        )


        st.dataframe(
            party_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # RECENT POSTS
    # ========================================================

    st.subheader(
        "🗣️ Recent Posts"
    )


    display_df = all_data[
        [
            "collected_at",
            "source",
            "sentiment_label",
            "sentiment_score",
            "party_mentioned",
            "issue_mentioned",
            "text"
        ]
    ].copy()


    display_df = (
        display_df
        .sort_values(
            "collected_at",
            ascending=False
        )
        .head(50)
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader(
        "📥 Download Data"
    )


    csv = all_data.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "Download CSV",
        csv,
        "swedish_election_sentiment.csv",
        "text/csv",
        use_container_width=True
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    init_database()

    show_dashboard()
