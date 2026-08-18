import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# Optional dependencies
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Swedish Election Sentiment 2026",
    page_icon="🇸🇪",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

DB_PATH = "swedish_election_2026.db"

ELECTION_DATE = datetime(2026, 9, 13)

SWEDISH_PARTIES = {
    "Socialdemokraterna": {
        "leader": "Magdalena Andersson",
        "short": "S",
        "bloc": "Left",
    },
    "Moderaterna": {
        "leader": "Ulf Kristersson",
        "short": "M",
        "bloc": "Right",
    },
    "Sverigedemokraterna": {
        "leader": "Jimmie Åkesson",
        "short": "SD",
        "bloc": "Right",
    },
    "Centerpartiet": {
        "leader": "Elisabeth Thand Ringqvist",
        "short": "C",
        "bloc": "Center",
    },
    "Vänsterpartiet": {
        "leader": "Nooshi Dadgostar",
        "short": "V",
        "bloc": "Left",
    },
    "Kristdemokraterna": {
        "leader": "Ebba Busch",
        "short": "KD",
        "bloc": "Right",
    },
    "Miljöpartiet": {
        "leader": "Amanda Lind / Daniel Helldén",
        "short": "MP",
        "bloc": "Green",
    },
    "Liberalerna": {
        "leader": "Simona Mohamsson",
        "short": "L",
        "bloc": "Center",
    },
}


SEARCH_TERMS = [
    "riksdagsval 2026",
    "svenska valet 2026",
    "valet 2026",
    "svensk politik",
    "regeringen",
    "Magdalena Andersson",
    "Ulf Kristersson",
    "Jimmie Åkesson",
    "Ebba Busch",
    "Nooshi Dadgostar",
    "Centerpartiet",
    "Liberalerna",
    "Miljöpartiet",
    "Socialdemokraterna",
    "Moderaterna",
    "Sverigedemokraterna",
    "Vänsterpartiet",
    "invandring",
    "kriminalitet",
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
    "svensk valdebatt 2026",
    "Magdalena Andersson",
    "Ulf Kristersson",
    "Jimmie Åkesson",
    "Ebba Busch",
    "Nooshi Dadgostar",
]


ISSUES = {
    "Immigration": [
        "invandring",
        "invandrare",
        "migration",
        "migrant",
        "asyl",
        "flykting",
        "integration",
    ],
    "Crime": [
        "kriminalitet",
        "brott",
        "gäng",
        "gängvåld",
        "skjutning",
        "polis",
        "fängelse",
        "straff",
    ],
    "Healthcare": [
        "sjukvård",
        "vård",
        "sjukhus",
        "läkare",
        "vårdcentral",
        "omsorg",
    ],
    "Education": [
        "skola",
        "skolan",
        "lärare",
        "utbildning",
        "betyg",
        "universitet",
    ],
    "Economy": [
        "ekonomi",
        "inflation",
        "skatt",
        "ränta",
        "jobb",
        "arbetslöshet",
        "lön",
        "löner",
    ],
    "Climate": [
        "klimat",
        "miljö",
        "utsläpp",
        "koldioxid",
        "global uppvärmning",
    ],
    "NATO": [
        "nato",
        "försvar",
        "militär",
        "ukraina",
        "försvarsmakten",
    ],
    "Housing": [
        "bostad",
        "bostäder",
        "hyra",
        "hyresrätt",
        "bolån",
        "bostadsmarknad",
    ],
    "Energy": [
        "energi",
        "elpris",
        "elpriser",
        "kärnkraft",
        "vindkraft",
        "el",
    ],
    "Welfare": [
        "välfärd",
        "bidrag",
        "pension",
        "socialförsäkring",
        "försäkringskassan",
    ],
}


# ============================================================
# SECRETS / ENVIRONMENT
# ============================================================

def get_secret(name, default=None):
    """
    Reads a secret from Streamlit secrets first,
    then environment variables.
    """

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


REDDIT_CLIENT_ID = get_secret("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = get_secret("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = get_secret(
    "REDDIT_USER_AGENT",
    "SwedishElectionSentimentMonitor/1.0",
)

YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY")


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
    )
    return conn


def init_database():

    conn = get_connection()
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

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS google_trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT,
            date TEXT,
            interest INTEGER,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


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
            "socialdemokrater",
            "sossarna",
            "magdalena andersson",
            "s-politik",
        ],
        "Moderaterna": [
            "moderaterna",
            "moderater",
            "ulf kristersson",
            "moderaternas",
        ],
        "Sverigedemokraterna": [
            "sverigedemokraterna",
            "sverigedemokrat",
            "sd",
            "jimmie åkesson",
        ],
        "Centerpartiet": [
            "centerpartiet",
            "centern",
            "centerpartiet",
        ],
        "Vänsterpartiet": [
            "vänsterpartiet",
            "vänstern",
            "nooshi dadgostar",
        ],
        "Kristdemokraterna": [
            "kristdemokraterna",
            "kristdemokrater",
            "kd",
            "ebba busch",
        ],
        "Miljöpartiet": [
            "miljöpartiet",
            "miljöpartiet de gröna",
            "mp",
        ],
        "Liberalerna": [
            "liberalerna",
            "liberal",
            "simona mohammson",
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

    for issue, keywords in ISSUES.items():

        for keyword in keywords:

            if keyword in text_lower:
                return issue

    return None


# ============================================================
# REDDIT
# ============================================================

def get_reddit_client():

    if praw is None:
        raise RuntimeError(
            "PRAW is not installed. Add praw to requirements.txt."
        )

    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise RuntimeError(
            "Reddit API credentials are missing."
        )

    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


def collect_reddit(limit=300):

    reddit = get_reddit_client()

    conn = get_connection()
    cursor = conn.cursor()

    collected = 0

    for subreddit_name in SUBREDDITS:

        try:

            subreddit = reddit.subreddit(subreddit_name)

            for post in subreddit.new(limit=limit):

                title = post.title or ""
                body = post.selftext or ""

                full_text = f"{title}\n{body}"

                party = detect_party(full_text)
                issue = detect_issue(full_text)

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO reddit_posts (
                        id,
                        source,
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
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post.id,
                        "reddit",
                        subreddit_name,
                        str(post.author) if post.author else None,
                        title,
                        body,
                        int(post.score or 0),
                        int(post.num_comments or 0),
                        float(post.created_utc),
                        post.url,
                        f"https://reddit.com{post.permalink}",
                        party,
                        issue,
                    ),
                )

                collected += 1

        except Exception as exc:

            st.warning(
                f"Reddit error in r/{subreddit_name}: {exc}"
            )

    conn.commit()
    conn.close()

    return collected


# ============================================================
# YOUTUBE
# ============================================================

def get_youtube_client():

    if build is None:
        raise RuntimeError(
            "Google API client is not installed. "
            "Add google-api-python-client to requirements.txt."
        )

    if not YOUTUBE_API_KEY:
        raise RuntimeError(
            "YOUTUBE_API_KEY is missing."
        )

    return build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY,
    )


def collect_youtube(max_results=30):

    youtube = get_youtube_client()

    conn = get_connection()
    cursor = conn.cursor()

    collected = 0

    for query in YOUTUBE_QUERIES:

        try:

            search_response = (
                youtube.search()
                .list(
                    q=query,
                    part="snippet",
                    type="video",
                    maxResults=max_results,
                    relevanceLanguage="sv",
                )
                .execute()
            )

            for item in search_response.get("items", []):

                video_id = item["id"]["videoId"]

                video_title = (
                    item["snippet"]
                    .get("title", "")
                )

                try:

                    comments_response = (
                        youtube.commentThreads()
                        .list(
                            part="snippet",
                            videoId=video_id,
                            maxResults=100,
                            textFormat="plainText",
                        )
                        .execute()
                    )

                except Exception:
                    continue

                for comment_item in comments_response.get(
                    "items",
                    []
                ):

                    snippet = (
                        comment_item["snippet"]
                        ["topLevelComment"]
                        ["snippet"]
                    )

                    comment_id = (
                        comment_item["id"]
                    )

                    text = snippet.get(
                        "textDisplay",
                        "",
                    )

                    party = detect_party(text)
                    issue = detect_issue(text)

                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO youtube_comments (
                            id,
                            source,
                            video_id,
                            video_title,
                            author,
                            text,
                            like_count,
                            published_at,
                            party_mentioned,
                            issue_mentioned
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            comment_id,
                            "youtube",
                            video_id,
                            video_title,
                            snippet.get(
                                "authorDisplayName"
                            ),
                            text,
                            int(
                                snippet.get(
                                    "likeCount",
                                    0,
                                )
                            ),
                            snippet.get(
                                "publishedAt"
                            ),
                            party,
                            issue,
                        ),
                    )

                    collected += 1

        except Exception as exc:

            st.warning(
                f"YouTube error for '{query}': {exc}"
            )

    conn.commit()
    conn.close()

    return collected


# ============================================================
# SENTIMENT MODEL
# ============================================================

@st.cache_resource
def load_sentiment_model():

    if pipeline is None:

        raise RuntimeError(
            "Transformers is not installed. "
            "Add transformers to requirements.txt."
        )

    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    )


def convert_sentiment(label):

    label_lower = label.lower()

    if label_lower in [
        "positive",
        "label_2",
    ]:
        return "positive"

    if label_lower in [
        "negative",
        "label_0",
    ]:
        return "negative"

    return "neutral"


def analyze_database():

    classifier = load_sentiment_model()

    conn = get_connection()
    cursor = conn.cursor()

    reddit_rows = cursor.execute(
        """
        SELECT id, title, text
        FROM reddit_posts
        WHERE sentiment_label IS NULL
        """
    ).fetchall()

    youtube_rows = cursor.execute(
        """
        SELECT id, text
        FROM youtube_comments
        WHERE sentiment_label IS NULL
        """
    ).fetchall()

    analyzed = 0

    # -----------------------------
    # Reddit
    # -----------------------------

    for row_id, title, text in reddit_rows:

        try:

            full_text = f"{title or ''}\n{text or ''}"

            result = classifier(
                full_text[:512]
            )[0]

            sentiment = convert_sentiment(
                result["label"]
            )

            score = float(
                result["score"]
            )

            cursor.execute(
                """
                UPDATE reddit_posts
                SET sentiment_label = ?,
                    sentiment_score = ?
                WHERE id = ?
                """,
                (
                    sentiment,
                    score,
                    row_id,
                ),
            )

            analyzed += 1

        except Exception as exc:

            st.warning(
                f"Sentiment error for Reddit post "
                f"{row_id}: {exc}"
            )

    # -----------------------------
    # YouTube
    # -----------------------------

    for row_id, text in youtube_rows:

        try:

            result = classifier(
                (text or "")[:512]
            )[0]

            sentiment = convert_sentiment(
                result["label"]
            )

            score = float(
                result["score"]
            )

            cursor.execute(
                """
                UPDATE youtube_comments
                SET sentiment_label = ?,
                    sentiment_score = ?
                WHERE id = ?
                """,
                (
                    sentiment,
                    score,
                    row_id,
                ),
            )

            analyzed += 1

        except Exception as exc:

            st.warning(
                f"Sentiment error for YouTube "
                f"comment {row_id}: {exc}"
            )

    conn.commit()
    conn.close()

    return analyzed


# ============================================================
# SUMMARY
# ============================================================

def generate_summary():

    conn = get_connection()

    query = """
        SELECT
            sentiment_label,
            sentiment_score
        FROM reddit_posts
        WHERE sentiment_label IS NOT NULL

        UNION ALL

        SELECT
            sentiment_label,
            sentiment_score
        FROM youtube_comments
        WHERE sentiment_label IS NOT NULL
    """

    df = pd.read_sql_query(
        query,
        conn,
    )

    if df.empty:

        conn.close()
        return

    total = len(df)

    positive = int(
        (df["sentiment_label"] == "positive").sum()
    )

    negative = int(
        (df["sentiment_label"] == "negative").sum()
    )

    neutral = int(
        (df["sentiment_label"] == "neutral").sum()
    )

    avg_sentiment = float(
        df["sentiment_score"].mean()
    )

    date_value = datetime.now().strftime(
        "%Y-%m-%d"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO sentiment_summary (
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
            date_value,
            total,
            positive,
            negative,
            neutral,
            avg_sentiment,
            "N/A",
            "N/A",
        ),
    )

    conn.commit()
    conn.close()


# ============================================================
# GOOGLE TRENDS
# ============================================================

def collect_google_trends():

    if TrendReq is None:

        raise RuntimeError(
            "pytrends is not installed. "
            "Add pytrends to requirements.txt."
        )

    pytrends = TrendReq(
        hl="sv-SE",
        tz=120,
    )

    trend_keywords = [
        "Socialdemokraterna",
        "Moderaterna",
        "Sverigedemokraterna",
        "Vänsterpartiet",
        "Centerpartiet",
        "Kristdemokraterna",
        "Miljöpartiet",
        "Liberalerna",
    ]

    conn = get_connection()

    collected = 0

    for i in range(
        0,
        len(trend_keywords),
        5,
    ):

        batch = trend_keywords[
            i:i + 5
        ]

        try:

            pytrends.build_payload(
                batch,
                timeframe="today 3-m",
                geo="SE",
                gprop="",
            )

            trends_df = (
                pytrends
                .interest_over_time()
            )

            if trends_df.empty:
                continue

            trends_df = trends_df.reset_index()

            for _, row in trends_df.iterrows():

                date_value = str(
                    row["date"]
                )

                for keyword in batch:

                    if keyword not in row:
                        continue

                    interest = int(
                        row[keyword]
                    )

                    conn.execute(
                        """
                        INSERT INTO google_trends (
                            keyword,
                            date,
                            interest
                        )
                        VALUES (?, ?, ?)
                        """,
                        (
                            keyword,
                            date_value,
                            interest,
                        ),
                    )

                    collected += 1

        except Exception as exc:

            st.warning(
                f"Google Trends error: {exc}"
            )

    conn.commit()
    conn.close()

    return collected


# ============================================================
# DASHBOARD DATA
# ============================================================

def load_all_sentiment_data():

    conn = get_connection()

    query = """
        SELECT
            'Reddit' AS source,
            id,
            title AS content,
            sentiment_label,
            sentiment_score,
            party_mentioned,
            issue_mentioned
        FROM reddit_posts

        UNION ALL

        SELECT
            'YouTube' AS source,
            id,
            text AS content,
            sentiment_label,
            sentiment_score,
            party_mentioned,
            issue_mentioned
        FROM youtube_comments
    """

    df = pd.read_sql_query(
        query,
        conn,
    )

    conn.close()

    return df


def load_trends():

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            keyword,
            date,
            interest
        FROM google_trends
        ORDER BY date
        """,
        conn,
    )

    conn.close()

    return df


# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard():

    st.title(
        "🇸🇪 Swedish Election Sentiment Monitor 2026"
    )

    days_until = (
        ELECTION_DATE - datetime.now()
    ).days

    st.subheader(
        f"🗳️ {max(days_until, 0)} days until the Swedish election"
    )

    # -----------------------------
    # Sidebar
    # -----------------------------

    st.sidebar.header(
        "⚙️ Data Collection"
    )

    if st.sidebar.button(
        "📥 Collect Reddit"
    ):

        with st.spinner(
            "Collecting Reddit data..."
        ):

            try:

                count = collect_reddit()

                st.sidebar.success(
                    f"Collected {count} Reddit posts."
                )

            except Exception as exc:

                st.sidebar.error(
                    str(exc)
                )

    if st.sidebar.button(
        "📺 Collect YouTube"
    ):

        with st.spinner(
            "Collecting YouTube comments..."
        ):

            try:

                count = collect_youtube()

                st.sidebar.success(
                    f"Collected {count} YouTube comments."
                )

            except Exception as exc:

                st.sidebar.error(
                    str(exc)
                )

    if st.sidebar.button(
        "📈 Collect Google Trends"
    ):

        with st.spinner(
            "Collecting Google Trends..."
        ):

            try:

                count = collect_google_trends()

                st.sidebar.success(
                    f"Collected {count} trend observations."
                )

            except Exception as exc:

                st.sidebar.error(
                    str(exc)
                )

    if st.sidebar.button(
        "🤖 Analyze Sentiment"
    ):

        with st.spinner(
            "Running AI sentiment analysis..."
        ):

            try:

                count = analyze_database()

                generate_summary()

                st.sidebar.success(
                    f"Analyzed {count} items."
                )

            except Exception as exc:

                st.sidebar.error(
                    str(exc)
                )

    if st.sidebar.button(
        "🔄 Refresh Dashboard"
    ):

        st.rerun()

    # -----------------------------
    # Party reference
    # -----------------------------

    with st.sidebar.expander(
        "🇸🇪 Party Reference"
    ):

        for party, data in SWEDISH_PARTIES.items():

            st.write(
                f"**{data['short']} — {party}**"
            )

            st.caption(
                data["leader"]
            )

    # -----------------------------
    # Load data
    # -----------------------------

    df = load_all_sentiment_data()

    if df.empty:

        st.info(
            "No data collected yet. "
            "Use the sidebar to collect Reddit, "
            "YouTube or Google Trends data."
        )

        return

    sentiment_df = df[
        df["sentiment_label"].notna()
    ].copy()

    # -----------------------------
    # Metrics
    # -----------------------------

    total_posts = len(df)

    if len(sentiment_df) > 0:

        positive_pct = (
            (
                sentiment_df["sentiment_label"]
                == "positive"
            ).mean()
            * 100
        )

        negative_pct = (
            (
                sentiment_df["sentiment_label"]
                == "negative"
            ).mean()
            * 100
        )

        avg_sentiment = (
            sentiment_df["sentiment_score"]
            .mean()
        )

    else:

        positive_pct = 0
        negative_pct = 0
        avg_sentiment = 0

    party_mentions = int(
        df["party_mentioned"]
        .notna()
        .sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Posts",
        total_posts,
    )

    col2.metric(
        "Positive %",
        f"{positive_pct:.1f}%",
    )

    col3.metric(
        "Negative %",
        f"{negative_pct:.1f}%",
    )

    col4.metric(
        "Party Mentions",
        party_mentions,
    )

    st.divider()

    # ========================================================
    # SENTIMENT DISTRIBUTION
    # ========================================================

    if not sentiment_df.empty:

        st.subheader(
            "📊 Sentiment Distribution"
        )

        sentiment_counts = (
            sentiment_df[
                "sentiment_label"
            ]
            .value_counts()
        )

        st.bar_chart(
            sentiment_counts
        )

        # -----------------------------
        # Source sentiment
        # -----------------------------

        st.subheader(
            "📡 Sentiment by Source"
        )

        source_sentiment = pd.crosstab(
            sentiment_df["source"],
            sentiment_df["sentiment_label"],
        )

        st.bar_chart(
            source_sentiment
        )

        # -----------------------------
        # Party sentiment
        # -----------------------------

        st.subheader(
            "🏛️ Sentiment by Party"
        )

        party_sentiment = pd.crosstab(
            sentiment_df[
                "party_mentioned"
            ],
            sentiment_df[
                "sentiment_label"
            ],
        )

        party_sentiment = (
            party_sentiment
            .drop(index=None, errors="ignore")
        )

        st.bar_chart(
            party_sentiment
        )

    # ========================================================
    # ISSUE ANALYSIS
    # ========================================================

    st.subheader(
        "🔥 Political Issues"
    )

    issue_counts = (
        df[
            df["issue_mentioned"]
            .notna()
        ]["issue_mentioned"]
        .value_counts()
    )

    if not issue_counts.empty:

        st.bar_chart(
            issue_counts
        )

    else:

        st.info(
            "No political issue data detected yet."
        )

    # ========================================================
    # GOOGLE TRENDS
    # ========================================================

    st.subheader(
        "📈 Google Search Trends"
    )

    trends_df = load_trends()

    if not trends_df.empty:

        trend_pivot = trends_df.pivot_table(
            index="date",
            columns="keyword",
            values="interest",
            aggfunc="mean",
        )

        st.line_chart(
            trend_pivot
        )

    else:

        st.info(
            "No Google Trends data collected yet."
        )

    # ========================================================
    # RECENT POSTS
    # ========================================================

    st.subheader(
        "📰 Recent Data"
    )

    display_columns = [
        "source",
        "content",
        "sentiment_label",
        "sentiment_score",
        "party_mentioned",
        "issue_mentioned",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    recent = df[
        available_columns
    ].tail(50)

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():

    init_database()

    show_dashboard()


if __name__ == "__main__":
    main()
