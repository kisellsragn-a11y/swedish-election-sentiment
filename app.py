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

import streamlit as st

# Hugging Face / Transformers
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud


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


# ============================================================
# SENTIMENT MODEL
# ============================================================

# IMPORTANT:
# We intentionally DO NOT use:
#
# cardiffnlp/twitter-xlm-roberta-base-sentiment
#
# because the tokenizer in that model is causing the
# sentencepiece/tiktoken parsing error on Streamlit Cloud.
#
# This multilingual DistilBERT model uses a normal tokenizer
# and supports multilingual sentiment analysis.

SENTIMENT_MODEL = (
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
)


# ============================================================
# SECRETS / API KEYS
# ============================================================

try:
    REDDIT_CLIENT_ID = st.secrets.get(
        "REDDIT_CLIENT_ID",
        os.environ.get(
            "REDDIT_CLIENT_ID",
            "YOUR_REDDIT_CLIENT_ID",
        ),
    )

    REDDIT_CLIENT_SECRET = st.secrets.get(
        "REDDIT_CLIENT_SECRET",
        os.environ.get(
            "REDDIT_CLIENT_SECRET",
            "YOUR_REDDIT_CLIENT_SECRET",
        ),
    )

    YOUTUBE_API_KEY = st.secrets.get(
        "YOUTUBE_API_KEY",
        os.environ.get(
            "YOUTUBE_API_KEY",
            "YOUR_YOUTUBE_API_KEY",
        ),
    )

except Exception:
    REDDIT_CLIENT_ID = os.environ.get(
        "REDDIT_CLIENT_ID",
        "YOUR_REDDIT_CLIENT_ID",
    )

    REDDIT_CLIENT_SECRET = os.environ.get(
        "REDDIT_CLIENT_SECRET",
        "YOUR_REDDIT_CLIENT_SECRET",
    )

    YOUTUBE_API_KEY = os.environ.get(
        "YOUTUBE_API_KEY",
        "YOUR_YOUTUBE_API_KEY",
    )


# ============================================================
# DATABASE
# ============================================================

DB_PATH = "swedish_election_2026.db"


def init_database():

    conn = sqlite3.connect(DB_PATH)

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
            like_count INTEGER,
            published_at TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            party_mentioned TEXT,
            issue_mentioned TEXT,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            avg_sentiment REAL,
            top_positive TEXT,
            top_negative TEXT
        )
        """
    )

    conn.commit()
    conn.close()


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
# PARTY DETECTION
# ============================================================

def detect_party(text):

    if not text:
        return None

    text_lower = text.lower()

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
            "jimmie åkesson",
            "jimmie akesson",
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
            "miljöparti",
            "miljoparti",
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


# ============================================================
# ISSUE DETECTION
# ============================================================

def detect_issue(text):

    if not text:
        return None

    text_lower = text.lower()

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
# REDDIT COLLECTOR
# ============================================================

def collect_reddit(limit=300):

    if REDDIT_CLIENT_ID == "YOUR_REDDIT_CLIENT_ID":

        return (
            0,
            "ERROR: Reddit API credentials are not configured.",
        )

    try:

        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent="SwedishElectionMonitor/1.0",
        )

    except Exception as e:

        return (
            0,
            f"ERROR: Reddit authentication failed: {e}",
        )

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    count = 0

    for subreddit_name in SUBREDDITS:

        try:

            subreddit = reddit.subreddit(subreddit_name)

        except Exception:

            continue

        for term in SEARCH_TERMS:

            try:

                limit_per_term = max(
                    1,
                    limit // len(SEARCH_TERMS),
                )

                posts = subreddit.search(
                    term,
                    limit=limit_per_term,
                    sort="new",
                )

                for post in posts:

                    title = post.title or ""
                    body = post.selftext or ""

                    combined_text = (
                        title + " " + body
                    )

                    party = detect_party(
                        combined_text
                    )

                    issue = detect_issue(
                        combined_text
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
                            issue_mentioned
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            post.id,
                            subreddit_name,
                            str(post.author)
                            if post.author
                            else "Unknown",
                            title,
                            body,
                            post.score,
                            post.num_comments,
                            post.created_utc,
                            post.url,
                            post.permalink,
                            party,
                            issue,
                        ),
                    )

                    count += 1

            except Exception:

                continue

    conn.commit()
    conn.close()

    return (
        count,
        f"Collected {count} Reddit posts",
    )


# ============================================================
# YOUTUBE COLLECTOR
# ============================================================

def collect_youtube(max_results=30):

    if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY":

        return (
            0,
            "ERROR: YouTube API key is not configured.",
        )

    try:

        youtube = build(
            "youtube",
            "v3",
            developerKey=YOUTUBE_API_KEY,
        )

    except Exception as e:

        return (
            0,
            f"ERROR: YouTube API initialization failed: {e}",
        )

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    count = 0

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

                video_title = video[
                    "snippet"
                ]["title"]

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

                        comment = (
                            item["snippet"]
                            ["topLevelComment"]
                            ["snippet"]
                        )

                        comment_id = item["id"]

                        text = (
                            comment["textDisplay"]
                            or ""
                        )

                        party = detect_party(text)

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
                                    "Unknown",
                                ),
                                text,
                                comment.get(
                                    "likeCount",
                                    0,
                                ),
                                comment.get(
                                    "publishedAt",
                                    "",
                                ),
                                party,
                                issue,
                            ),
                        )

                        count += 1

                except Exception:

                    continue

        except Exception:

            continue

    conn.commit()
    conn.close()

    return (
        count,
        f"Collected {count} YouTube comments",
    )


# ============================================================
# LOAD SENTIMENT MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_model():

    tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL
    )

    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1,
    )

    return classifier


# ============================================================
# SENTIMENT NORMALIZATION
# ============================================================

def normalize_sentiment(label, score):

    label = str(label).lower()

    if "positive" in label:

        return "positive", float(score)

    if "negative" in label:

        return "negative", -float(score)

    if "neutral" in label:

        return "neutral", 0.0

    # Some models may return labels such as LABEL_0.
    # We make a safe fallback instead of crashing.

    return "neutral", 0.0


# ============================================================
# ANALYZE DATABASE
# ============================================================

def analyze_database():

    try:

        classifier = load_sentiment_model()

    except Exception as e:

        return (
            0,
            f"Could not load the sentiment model: {e}",
        )

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    analyzed_count = 0

    # --------------------------------------------------------
    # REDDIT
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT id, title, text
        FROM reddit_posts
        WHERE sentiment_label IS NULL
        """
    )

    reddit_posts = cursor.fetchall()

    for post_id, title, text in reddit_posts:

        full_text = (
            f"{title or ''} {text or ''}"
        ).strip()

        if not full_text:

            label = "neutral"
            score = 0.0

        else:

            try:

                result = classifier(
                    full_text[:512]
                )[0]

                label, score = normalize_sentiment(
                    result["label"],
                    result["score"],
                )

            except Exception:

                label = "neutral"
                score = 0.0

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

        analyzed_count += 1

    # --------------------------------------------------------
    # YOUTUBE
    # --------------------------------------------------------

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
                    result["label"],
                    result["score"],
                )

            except Exception:

                label = "neutral"
                score = 0.0

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

        analyzed_count += 1

    conn.commit()
    conn.close()

    return (
        analyzed_count,
        "Sentiment analysis completed successfully.",
    )


# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary():

    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            sentiment_label,
            sentiment_score,
            text,
            party_mentioned,
            issue_mentioned,
            'reddit' AS source
        FROM reddit_posts
        WHERE sentiment_label IS NOT NULL

        UNION ALL

        SELECT
            sentiment_label,
            sentiment_score,
            text,
            party_mentioned,
            issue_mentioned,
            'youtube' AS source
        FROM youtube_comments
        WHERE sentiment_label IS NOT NULL
    """

    df = pd.read_sql_query(
        query,
        conn,
    )

    conn.close()

    if len(df) == 0:

        return None

    summary = {

        "date": datetime.now().strftime(
            "%Y-%m-%d"
        ),

        "total_posts": len(df),

        "positive_count": len(
            df[
                df["sentiment_label"]
                == "positive"
            ]
        ),

        "negative_count": len(
            df[
                df["sentiment_label"]
                == "negative"
            ]
        ),

        "neutral_count": len(
            df[
                df["sentiment_label"]
                == "neutral"
            ]
        ),

        "avg_sentiment": float(
            df["sentiment_score"].mean()
        ),
    }

    conn = sqlite3.connect(DB_PATH)

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
            "N/A",
        ),
    )

    conn.commit()
    conn.close()

    return summary


# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

def load_dashboard_data():

    conn = sqlite3.connect(DB_PATH)

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

    days_until = (
        election_date - datetime.now()
    ).days

    if days_until < 0:

        days_until = 0

    st.markdown(
        f"""
        **Riksdagsval: September 13, 2026**
        | **{days_until} days until election**
        """
    )

    st.markdown(
        """
        AI-powered social media sentiment
        analysis for the Swedish general election.
        """
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.title("🎛️ Controls")

        st.markdown(
            "### Data Collection"
        )

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

            if reddit_count > 0:

                st.success(
                    reddit_msg
                )

            else:

                st.warning(
                    reddit_msg
                )

            if youtube_count > 0:

                st.success(
                    youtube_msg
                )

            else:

                st.warning(
                    youtube_msg
                )

        st.markdown("---")

        st.markdown(
            "### AI Analysis"
        )

        if st.button(
            "🧠 Analyze Sentiment",
            use_container_width=True,
        ):

            with st.spinner(
                "Loading AI sentiment model..."
            ):

                analyzed, message = (
                    analyze_database()
                )

            if analyzed > 0:

                generate_summary()

                st.success(
                    f"Analyzed {analyzed} items"
                )

            else:

                st.error(
                    message
                )

        if st.button(
            "📊 Refresh Dashboard",
            use_container_width=True,
        ):

            st.rerun()

        st.markdown("---")

        st.subheader(
            "🤖 AI Model"
        )

        st.caption(
            SENTIMENT_MODEL
        )

        st.markdown("---")

        st.subheader(
            "🇸🇪 Party Reference"
        )

        for party, info in SWEDISH_PARTIES.items():

            st.markdown(
                f"""
                **{info['abbrev']}** —
                {party}
                """
            )

    # ========================================================
    # LOAD DATA
    # ========================================================

    (
        reddit_df,
        youtube_df,
        summary_df,
    ) = load_dashboard_data()

    # ========================================================
    # PREPARE DATA
    # ========================================================

    if len(reddit_df) > 0:

        reddit_df["source"] = "Reddit"

        reddit_df["text"] = (
            reddit_df["title"].fillna("")
            + " "
            + reddit_df["text"].fillna("")
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

    if len(youtube_df) > 0:

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
    # NO DATA
    # ========================================================

    if len(all_data) == 0:

        st.info(
            """
            No analyzed data yet.

            1. Click **Collect New Data**
            2. Then click **Analyze Sentiment**
            3. The dashboard will populate automatically.
            """
        )

        return

    # ========================================================
    # TOP METRICS
    # ========================================================

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(
            "Total Posts",
            len(all_data),
        )

    with col2:

        positive_count = len(
            all_data[
                all_data["sentiment_label"]
                == "positive"
            ]
        )

        positive_pct = (
            positive_count
            / len(all_data)
            * 100
        )

        st.metric(
            "Positive %",
            f"{positive_pct:.1f}%",
        )

    with col3:

        negative_count = len(
            all_data[
                all_data["sentiment_label"]
                == "negative"
            ]
        )

        negative_pct = (
            negative_count
            / len(all_data)
            * 100
        )

        st.metric(
            "Negative %",
            f"{negative_pct:.1f}%",
        )

    with col4:

        avg_score = all_data[
            "sentiment_score"
        ].mean()

        st.metric(
            "Avg Sentiment",
            f"{avg_score:.3f}",
        )

    with col5:

        total_parties = (
            all_data[
                "party_mentioned"
            ]
            .notna()
            .sum()
        )

        st.metric(
            "Party Mentions",
            total_parties,
        )

    # ========================================================
    # SENTIMENT DISTRIBUTION
    # ========================================================

    col_left, col_right = st.columns(2)

    with col_left:

        st.subheader(
            "📊 Sentiment Distribution"
        )

        sentiment_counts = (
            all_data[
                "sentiment_label"
            ]
            .value_counts()
        )

        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
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

        st.subheader(
            "📱 Sentiment by Source"
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
    # PARTY + ISSUES
    # ========================================================

    col_left2, col_right2 = (
        st.columns(2)
    )

    with col_left2:

        st.subheader(
            "🇸🇪 Sentiment by Party"
        )

        party_data = all_data[
            all_data[
                "party_mentioned"
            ].notna()
        ]

        if len(party_data) > 0:

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
                color_discrete_map={
                    "positive": "#2ecc71",
                    "negative": "#e74c3c",
                    "neutral": "#95a5a6",
                },
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

    with col_right2:

        st.subheader(
            "🔥 Top Issues Discussed"
        )

        issue_data = all_data[
            all_data[
                "issue_mentioned"
            ].notna()
        ]

        if len(issue_data) > 0:

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

    if len(summary_df) > 0:

        summary_df = summary_df.sort_values(
            "date"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=summary_df["date"],
                y=summary_df[
                    "avg_sentiment"
                ],
                mode="lines+markers",
                name="Average Sentiment",
                line=dict(
                    color="#3498db",
                    width=3,
                ),
            )
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
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
            "Sentiment trend will appear after analysis."
        )

    # ========================================================
    # WORD CLOUDS
    # ========================================================

    col_wc1, col_wc2 = (
        st.columns(2)
    )

    with col_wc1:

        st.subheader(
            "🔴 Word Cloud — Negative Posts"
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

            try:

                wordcloud = WordCloud(
                    width=600,
                    height=400,
                    background_color="white",
                    colormap="Reds",
                ).generate(
                    negative_text
                )

                st.image(
                    wordcloud.to_array(),
                    use_container_width=True,
                )

            except Exception:

                st.info(
                    "Not enough text for word cloud."
                )

        else:

            st.info(
                "No negative text available."
            )

    with col_wc2:

        st.subheader(
            "🟢 Word Cloud — Positive Posts"
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

            try:

                wordcloud = WordCloud(
                    width=600,
                    height=400,
                    background_color="white",
                    colormap="Greens",
                ).generate(
                    positive_text
                )

                st.image(
                    wordcloud.to_array(),
                    use_container_width=True,
                )

            except Exception:

                st.info(
                    "Not enough text for word cloud."
                )

        else:

            st.info(
                "No positive text available."
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

    display_df = display_df.sort_values(
        "collected_at",
        ascending=False,
    ).head(50)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader(
        "💾 Download Data"
    )

    csv = all_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        csv,
        "swedish_election_sentiment.csv",
        "text/csv",
        use_container_width=True,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    init_database()

    show_dashboard()
