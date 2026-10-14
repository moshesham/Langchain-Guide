import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time

# Page config
st.set_page_config(
    page_title="Meta DE Prep Blueprint",
    layout="wide",
    page_icon="⚡"
)

# Meta Color Palette & Constants
META_BLUE = "#0666FE"
META_BLUE_HOVER = "#0550c9"
META_TEAL = "#00B4D8"
META_PURPLE = "#8B5CF6"
BG_DARK = "#0F172A"
BG_CARD = "#1E293B"
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"
BORDER_DARK = "#334155"

# Safe pandas styler helper for cross-version compatibility
def style_dataframe_safe(df, highlight_col, condition_fn, color="#047857"):
    try:
        def cell_styler(val):
            return f"background-color: {color}; color: #FFFFFF;" if condition_fn(val) else ""
        
        if hasattr(df.style, "map"):
            return df.style.map(cell_styler, subset=[highlight_col])
        elif hasattr(df.style, "applymap"):
            return df.style.applymap(cell_styler, subset=[highlight_col])
    except Exception:
        pass
    return df

# Initialize session state for persistent tracking
if "checklist_state" not in st.session_state:
    st.session_state.checklist_state = {
        "SQL Window Functions (Dense_Rank, Row_Number)": True,
        "Python Interval Merging & Stream Deque": True,
        "Star Schema vs Snowflake Trade-offs": True,
        "SCD Type 2 Dimension Implementation": False,
        "Presto / Spark Shuffle & Salting for Data Skew": False,
        "Broadcast vs Hash Joins & Partition Pruning": False,
        "Idempotent Backfill & Circuit Breakers": False,
        "Product Sense: Root Cause Analysis (RCA) Framework": True,
        "Feature Store & Training-Serving Skew Monitoring": False,
        "Cohort Retention Triangles & Funnel Conversions": False,
        "Write 5 Meta Core Values STAR Stories": False,
        "Complete 2 Full 45-min Timed Mock Screens": False,
    }

if "star_stories" not in st.session_state:
    st.session_state.star_stories = [
        {
            "value": "Focus on Long-Term Impact",
            "title": "Migrated Petabyte-Scale User Engagement Pipeline to Incremental Merge",
            "summary": "Reduced pipeline compute costs by 42% and SLA runtime from 6.5 hours to 45 minutes by implementing hourly incremental delta merges with partition swapping."
        }
    ]

# ---------------------------------------------------------
# DASHBOARD / OVERVIEW
# ---------------------------------------------------------
def render_dashboard():
    st.title("🏠 Meta Data Engineer Prep Blueprint")
    st.caption("Interactive Technical, Data Modeling, Distributed Architecture & Case Study Guide")
    
    # Header KPI stats
    total_tasks = len(st.session_state.checklist_state)
    done_tasks = sum(1 for v in st.session_state.checklist_state.values() if v)
    readiness_pct = int((done_tasks / total_tasks) * 100) if total_tasks else 0
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="card">
            <div class="label">Interview Funnel Pass Rate</div>
            <div style="font-size: 22px; font-weight: 700; color: {META_BLUE}; margin: 4px 0;">2.4%</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">Top 2% bar for L4/L5 DE offers</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="card">
            <div class="label">Checklist Readiness</div>
            <div style="font-size: 22px; font-weight: 700; color: #10B981; margin: 4px 0;">{readiness_pct}%</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">{done_tasks}/{total_tasks} core competencies validated</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="card">
            <div class="label">Onsite Evaluation Rounds</div>
            <div style="font-size: 22px; font-weight: 700; color: #F59E0B; margin: 4px 0;">5 Rounds</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">2x Tech Screen, 2x System/Modeling, 1x Behavioral</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="card">
            <div class="label">Core Architecture Stack</div>
            <div style="font-size: 22px; font-weight: 700; color: #A855F7; margin: 4px 0;">Presto / Spark</div>
            <div style="font-size: 11px; color: {TEXT_MUTED};">Trino, Flink, RocksDB, Iceberg/Hive</div>
        </div>
        """, unsafe_allow_html=True)

    col_funnel, col_radar = st.columns([5, 7])
    
    with col_funnel:
        st.markdown("### Interview Funnel Lifecycle")
        stages = [
            "Recruiter Screen (100%)",
            "Technical Screen: SQL + Python (40%)",
            "Data Modeling Case (18%)",
            "System Architecture & Big Data Case (8%)",
            "Behavioral & Culture Fit (4%)",
            "Final Offer Extends (2.4%)"
        ]
        rates = [100, 40, 18, 8, 4, 2.4]
        colors = [META_BLUE, "#38BDF8", "#818CF8", "#A78BFA", "#C084FC", "#10B981"]
        
        fig_funnel = go.Figure(go.Funnel(
            y=stages,
            x=rates,
            textinfo="value+percent initial",
            marker=dict(color=colors)
        ))
        fig_funnel.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_LIGHT, size=11),
            height=300
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_radar:
        st.markdown("### Competency Domains Assessment")
        categories = ['SQL Engine Optimization', 'Python Algorithms', 'Dimensional Modeling', 'Distributed Pipelines', 'Product Sense / RCA', 'System Design']
        
        with st.expander("⚙️ Adjust Competency Skill Ratings", expanded=False):
            c_cols = st.columns(3)
            user_ratings = []
            for idx, cat in enumerate(categories):
                with c_cols[idx % 3]:
                    val = st.slider(cat, 1, 5, 4 if idx < 3 else 3, key=f"radar_{idx}")
                    user_ratings.append(val)
        
        if 'user_ratings' not in locals():
            user_ratings = [4, 4, 4, 3, 3, 3]

        meta_target = [5, 4.5, 4.5, 4, 4.5, 4]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_ratings + [user_ratings[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Your Current Level',
            line=dict(color=META_BLUE, width=2),
            fillcolor='rgba(6, 102, 254, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=meta_target + [meta_target[0]],
            theta=categories + [categories[0]],
            name='Meta Target (L5 Senior DE)',
            line=dict(color='#10B981', dash='dash', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5], color=TEXT_MUTED, gridcolor=BORDER_DARK),
                angularaxis=dict(color=TEXT_LIGHT, gridcolor=BORDER_DARK)
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=10, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_LIGHT, size=11),
            height=300
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Quick Module Access")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("""
        **Core Data Architecture**
        - [Module 2] Star Schema & SCD Type 2
        - [Module 3] Presto/Trino SQL & Windowing
        - [Module 5] Atomic Partition Swaps & DAGs
        """)
    with m_col2:
        st.markdown("""
        **Product & Algorithms**
        - [Module 1] Metric Drop RCA Simulator
        - [Module 4] Interval Merging & Stream Queues
        - [Module 7] Funnels & Retention Triangles
        """)
    with m_col3:
        st.markdown("""
        **Infrastructure & Interview**
        - [Module 6] AI Feature Stores & Drift
        - [Module 8] Authentic Meta Core Values STAR
        - [Module 9] 4-Week Action Roadmap
        """)

# ---------------------------------------------------------
# MODULE 1: PRODUCT SENSE & METRICS STRATEGY
# ---------------------------------------------------------
def render_module_1():
    st.title("📊 Module 1: Product Sense & RCA (Root Cause Analysis)")
    st.caption("Master Meta's Framework for Diagnosing Complex Metric Drops & Building Metric Trees")
    
    st.markdown("""
    <div class="card">
        <div class="label">Meta RCA 6-Step Interview Framework</div>
        <ol style="margin-top: 6px; padding-left: 20px; font-size: 12px; color: #E2E8F0; line-height: 1.6;">
            <li><strong>Clarify the Metric:</strong> Exact definition, numerator/denominator, time horizon, and baseline seasonality.</li>
            <li><strong>Verify Data Pipeline Integrity:</strong> Is it a true user behavior drop or a logging/ETL/telemetry outage?</li>
            <li><strong>Segment Dimensions:</strong> Slice by OS, App Version, Geography, Network Type, User Tenure, and Traffic Source.</li>
            <li><strong>Isolate Anomaly:</strong> Pinpoint the single dimension or intersection where the delta is concentrated.</li>
            <li><strong>Form Hypotheses:</strong> App releases, feature rollouts, third-party API dependencies, cannibalization, or external holidays/blackouts.</li>
            <li><strong>Action & Mitigation:</strong> Feature flag rollback, hotfix deployment, pipeline backfill, and executive communication.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Interactive RCA Metric Drop Simulator")
    selected_scenario = st.selectbox(
        "Select Active Outage Case Study:",
        [
            "Scenario A: Instagram Reels Watch Time Drop (-8.2%)",
            "Scenario B: Facebook Active Video Creators Decline (-6.4%)"
        ]
    )
    
    col_filters1, col_filters2 = st.columns(2)
    with col_filters1:
        os_filter = st.multiselect("OS Filter", ["iOS", "Android", "Web"], default=["iOS", "Android", "Web"])
        region_filter = st.multiselect("Region", ["NA", "EMEA", "APAC", "LATAM"], default=["NA", "EMEA", "APAC", "LATAM"])
    with col_filters2:
        tenure_filter = st.multiselect("User Cohort Tenure", ["New (<30d)", "Active Core", "Resurrected"], default=["New (<30d)", "Active Core", "Resurrected"])
        version_filter = st.multiselect("App Version", ["v200 (Stable)", "v201 (Rollout)", "v202 (Latest Canary)"], default=["v200 (Stable)", "v201 (Rollout)", "v202 (Latest Canary)"])
        
    st.write("---")
    
    if "Scenario A" in selected_scenario:
        # Condition: iOS + v202 + NA/EMEA
        is_isolated = ("iOS" in os_filter and "v202 (Latest Canary)" in version_filter and len(os_filter) == 1 and len(version_filter) == 1)
        contains_bug = ("iOS" in os_filter and "v202 (Latest Canary)" in version_filter)
        
        if is_isolated:
            st.error("🚨 ROOT CAUSE CONFIRMED: -36.4% Watch Time crash strictly isolated to iOS v202 in NA & EMEA! A deadlock bug in the new AVPlayer hardware video decoder causes autoplay stalls on high-refresh ProMotion screens.")
        elif contains_bug:
            st.warning("⚠️ High variance detected: iOS v202 is exhibiting heavy drops, but your filters still include unaffected platforms. Narrow down to isolate.")
        else:
            st.success("✅ Metric looks stable across the selected sub-dimensions. Keep slicing across OS and Version releases.")
            
        dates = pd.date_range(end="2024-05-14", periods=14)
        df_chart = pd.DataFrame({
            "Date": dates,
            "iOS v202 (Defective)": [100, 101, 99, 102, 100, 98, 64, 62, 63, 61, 60, 62, 59, 61],
            "iOS v201 (Stable)": [100, 101, 100, 102, 101, 99, 100, 101, 99, 100, 101, 99, 100, 101],
            "Android v202": [99, 100, 101, 100, 99, 100, 101, 100, 99, 100, 101, 100, 101, 100]
        })
        fig = px.line(df_chart, x="Date", y=["iOS v202 (Defective)", "iOS v201 (Stable)", "Android v202"], title="Watch Time Index by Platform & Version (Release Date: May 7)")
        fig.add_vline(x="2024-05-07", line_width=2, line_dash="dash", line_color="#EF4444", annotation_text="v202 Canary Release")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Scenario B: Android + APAC + v201
        is_isolated = ("Android" in os_filter and "APAC" in region_filter and "v201 (Rollout)" in version_filter and len(os_filter) == 1 and len(region_filter) == 1)
        contains_bug = ("Android" in os_filter and "APAC" in region_filter and "v201 (Rollout)" in version_filter)
        
        if is_isolated:
            st.error("🚨 ROOT CAUSE CONFIRMED: -29.2% Active Video Creators in Android v201 across APAC! Audio picker API calls are timing out due to a regional copyright CDN certificate failure during the upload flow.")
        elif contains_bug:
            st.warning("⚠️ Anomaly signals detected in Android APAC traffic. Refine your filters to pinpoint the exact version and cohort.")
        else:
            st.success("✅ Metric looks stable across the selected dimensions. Investigate geographic network logs or specific platform versions.")
            
        regions = ["NA", "EMEA", "APAC", "LATAM"]
        df_region = pd.DataFrame({
            "Region": regions,
            "Creator Change %": [+1.2, -0.4, -29.2, +0.8]
        })
        fig = px.bar(df_region, x="Region", y="Creator Change %", color="Creator Change %", color_continuous_scale=["#EF4444", "#10B981"], title="Creator Delta by Region (Android)")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Metric Hierarchy & Guardrail Trade-offs")
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown("""
        <div class="card">
            <div class="label" style="color: #60A5FA;">L1: North Star Metric</div>
            <div style="font-weight: 700; font-size: 14px; margin: 6px 0;">Daily Active Engaged Users (DAU)</div>
            <p style="font-size: 11px; color: #94A3B8;">Represents core platform health and network value realization.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_l2:
        st.markdown("""
        <div class="card">
            <div class="label" style="color: #34D399;">L2: Input Driver Metrics</div>
            <div style="font-weight: 700; font-size: 14px; margin: 6px 0;">Video Watch Time, Creator Uploads, Session Length</div>
            <p style="font-size: 11px; color: #94A3B8;">Direct levers that engineering and product teams can optimize.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_l3:
        st.markdown("""
        <div class="card">
            <div class="label" style="color: #F87171;">Counter & Guardrail Metrics</div>
            <div style="font-weight: 700; font-size: 14px; margin: 6px 0;">Ad Hide Rate, App Crash Rate, Push Unsubscribe %</div>
            <p style="font-size: 11px; color: #94A3B8;">Ensures short-term engagement wins do not erode user trust or retention.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 2: DATA MODELING & SCHEMA DESIGN
# ---------------------------------------------------------
def render_module_2():
    st.title("📐 Module 2: Data Modeling & Schema Design")
    st.caption("Meta Dimensional Modeling: Fact Grains, SCD Type 2 Timelines & Cumulative Bitmasks")
    
    st.markdown("### Interactive Star Schema Explorer")
    selected_fact = st.selectbox(
        "Select Core Fact Table Grain:",
        ["fact_reel_impressions (Atomic Event Grain)", "fact_daily_creator_aggregates (Daily Rollup Grain)", "fact_ad_conversions (Transaction Grain)"]
    )
    
    if "fact_reel_impressions" in selected_fact:
        st.code("""
-- Fact Grain: 1 row per unique video playback event
CREATE TABLE fact_reel_impressions (
    impression_id       VARCHAR(64),      -- High cardinality event UUID
    user_id             BIGINT,           -- FK -> dim_user [PK]
    reel_id             BIGINT,           -- FK -> dim_reel [PK]
    creator_id          BIGINT,           -- FK -> dim_creator [PK]
    device_id           VARCHAR(64),      -- FK -> dim_device
    watch_duration_ms   INT,              -- Additive metric: ms watched
    video_length_ms     INT,              -- Additive metric: total duration
    is_completed        BOOLEAN,          -- Degenerate dimension: 100% watched
    is_liked            BOOLEAN,          -- Degenerate dimension: like event
    is_shared           BOOLEAN,          -- Degenerate dimension: share event
    ds                  VARCHAR(10)       -- Partition key: YYYY-MM-DD
)
PARTITIONED BY (ds)
CLUSTERED BY (user_id) INTO 256 BUCKETS;
        """, language="sql")
    elif "fact_daily_creator_aggregates" in selected_fact:
        st.code("""
-- Fact Grain: 1 row per creator per day
CREATE TABLE fact_daily_creator_aggregates (
    creator_id          BIGINT,           -- FK -> dim_creator [PK]
    ds                  VARCHAR(10),      -- Partition key: YYYY-MM-DD
    total_views         BIGINT,           -- Daily aggregate metric
    total_watch_sec     DOUBLE,           -- Daily aggregate metric
    net_new_followers   INT,              -- Daily delta
    unique_viewers      BIGINT            -- HyperLogLog sketch or count
)
PARTITIONED BY (ds);
        """, language="sql")
    else:
        st.code("""
-- Fact Grain: 1 row per ad click/conversion attributed event
CREATE TABLE fact_ad_conversions (
    conversion_id       VARCHAR(64),      -- UUID
    campaign_id         BIGINT,           -- FK -> dim_ad_campaign
    advertiser_id       BIGINT,           -- FK -> dim_advertiser
    user_id             BIGINT,           -- FK -> dim_user
    conversion_type     VARCHAR(32),      -- 'purchase', 'signup', 'lead'
    conversion_value    DECIMAL(12,2),    -- Attributed revenue
    attribution_window  VARCHAR(16),      -- '1d_click', '7d_view'
    ds                  VARCHAR(10)       -- Partition key: YYYY-MM-DD
)
PARTITIONED BY (ds);
        """, language="sql")

    st.markdown("---")
    st.markdown("### SCD Type 2 Timeline Simulator (Slowly Changing Dimensions)")
    st.write("Track dimension changes across time while preserving historical accuracy for point-in-time joins.")
    
    col_scd_in1, col_scd_in2, col_scd_in3 = st.columns(3)
    with col_scd_in1:
        new_loc = st.selectbox("Simulate User Relocation:", ["US -> UK", "UK -> Germany (DE)", "DE -> Japan (JP)"])
    with col_scd_in2:
        effective_date = st.date_input("Effective Transition Date", value=pd.to_datetime("2024-04-01"))
    with col_scd_in3:
        st.write("")
        st.write("")
        simulate_update = st.button("Apply SCD Type 2 Update", type="primary")

    df_scd = pd.DataFrame({
        "surrogate_key": [1001, 1002, 1003],
        "user_id": [884102, 884102, 884102],
        "country": ["US", "UK", "DE"],
        "tier": ["Standard", "Creator", "Creator VIP"],
        "effective_start_ds": ["2022-01-01", "2023-06-15", str(effective_date)],
        "effective_end_ds": ["2023-06-14", "2024-03-31", "9999-12-31"],
        "is_current": [False, False, True]
    })
    
    st.caption("Active Dimension Table (`dim_user_scd2`):")
    st.dataframe(style_dataframe_safe(df_scd, "is_current", lambda x: x == True, "#047857"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Meta Cumulative Retention Array & Bitmask Pattern")
    st.markdown("""
    At Meta scale (3+ billion users), performing 30-day self-joins on raw event tables is computationally prohibitive. 
    Instead, Meta stores a single integer or 32-bit bitmask per user in a daily cumulated table.
    """)
    
    col_bm1, col_bm2 = st.columns([6, 6])
    with col_bm1:
        st.markdown("**Interactive Bitmask Calculator:**")
        days_active = st.multiselect(
            "Select Days User Logged In (Last 7 Days):",
            [f"Day {i}" for i in range(1, 8)],
            default=["Day 1", "Day 2", "Day 3", "Day 7"]
        )
        
        # Calculate bitmask
        bitmask = 0
        for d in days_active:
            day_num = int(d.split()[1])
            bitmask |= (1 << (day_num - 1))
            
        binary_repr = f"{bitmask:07b}"
        l7_count = bin(bitmask & 0b1111111).count("1")
        is_active_today = bool(bitmask & 1)
        
        st.markdown(f"""
        - **Calculated Bitmask (Decimal):** `{bitmask}`
        - **Binary Representation:** `{binary_repr}`
        - **L7 Active Days:** `{l7_count} / 7 days`
        - **Active Today (D0):** `{'YES (Retained)' if is_active_today else 'NO (Inactive Today)'}`
        """)
    
    with col_bm2:
        st.code("""
-- Cumulative Table Schema
CREATE TABLE user_activity_cumulated (
    user_id             BIGINT,
    ds                  VARCHAR(10),     -- Today's partition
    dates_active        ARRAY<VARCHAR>,  -- Chronological login dates
    active_days_bitmask BIGINT           -- Bitwise representation
)
PARTITIONED BY (ds);

-- Bitwise Query for 7-Day Active Users (L7):
SELECT COUNT(user_id) AS l7_dau
FROM user_activity_cumulated
WHERE ds = '2024-05-14'
  AND (active_days_bitmask & 127) > 0; -- 127 is binary 1111111
        """, language="sql")

# ---------------------------------------------------------
# MODULE 3: SQL ENGINE EXECUTION & OPTIMIZATION
# ---------------------------------------------------------
def render_module_3():
    st.title("⚡ Module 3: SQL Engine Execution & Optimization")
    st.caption("Deep-Dive on Presto/Trino, Window Functions, MapReduce Shuffle & Join Algorithms")
    
    st.markdown("### Interactive Meta SQL Interview Problem Bank")
    selected_problem = st.selectbox(
        "Choose an Interview Question:",
        [
            "1. User Sessionization with 30-Minute Inactivity Window",
            "2. Top 3 Trending Reels per Country (DENSE_RANK)",
            "3. 7-Day Rolling Active Watch Duration (ROWS BETWEEN)",
            "4. Growth Accounting: User Lifecycle Transitions"
        ]
    )
    
    if "Sessionization" in selected_problem:
        st.markdown("""
        **Problem Statement:** Given a stream of user click events, group them into sessions. 
        A new session starts when a user has been inactive for **more than 30 minutes**.
        """)
        
        st.code("""
WITH event_gaps AS (
    SELECT 
        user_id,
        event_time,
        LAG(event_time) OVER (
            PARTITION BY user_id 
            ORDER BY event_time
        ) AS prev_event_time
    FROM user_click_events
),
session_flags AS (
    SELECT 
        user_id,
        event_time,
        CASE 
            WHEN prev_event_time IS NULL THEN 1
            WHEN DATE_DIFF('minute', prev_event_time, event_time) > 30 THEN 1
            ELSE 0 
        END AS is_new_session
    FROM event_gaps
)
SELECT 
    user_id,
    event_time,
    -- Cumulative sum of flags generates sequential session IDs
    SUM(is_new_session) OVER (
        PARTITION BY user_id 
        ORDER BY event_time 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS session_id
FROM session_flags;
        """, language="sql")
        
        # Sample evaluation
        df_demo = pd.DataFrame({
            "user_id": [101, 101, 101, 101, 102],
            "event_time": ["10:00", "10:15", "10:55", "11:05", "12:00"],
            "prev_time": [None, "10:00", "10:15", "10:55", None],
            "diff_min": [None, 15, 40, 10, None],
            "is_new_session": [1, 0, 1, 0, 1],
            "session_id": [1, 1, 2, 2, 1]
        })
        st.caption("Live Pipeline Output:")
        st.dataframe(df_demo, use_container_width=True)

    elif "DENSE_RANK" in selected_problem:
        st.markdown("""
        **Problem Statement:** Find the Top 3 most-watched Reels in each country. If there are ties in view count, they should share the same rank without skipping consecutive ranks.
        """)
        st.code("""
WITH ranked_reels AS (
    SELECT 
        country,
        reel_id,
        view_count,
        DENSE_RANK() OVER (
            PARTITION BY country 
            ORDER BY view_count DESC
        ) AS rank_num
    FROM reel_country_daily_metrics
    WHERE ds = '2024-05-14'
)
SELECT country, reel_id, view_count, rank_num
FROM ranked_reels
WHERE rank_num <= 3
ORDER BY country, rank_num;
        """, language="sql")
        
        df_rank = pd.DataFrame({
            "country": ["US", "US", "US", "US", "UK", "UK"],
            "reel_id": [801, 805, 809, 812, 901, 902],
            "view_count": [1500000, 1200000, 1200000, 950000, 800000, 750000],
            "rank_num": [1, 2, 2, 3, 1, 2]
        })
        st.caption("Rank Output (Note DENSE_RANK preserves ties):")
        st.dataframe(df_rank, use_container_width=True)

    elif "ROWS BETWEEN" in selected_problem:
        st.markdown("""
        **Problem Statement:** Compute the 7-day rolling average watch duration for each creator, ordering by partition date `ds`.
        """)
        st.code("""
SELECT 
    creator_id,
    ds,
    daily_watch_sec,
    -- Physical window frame of 6 preceding rows + current row
    AVG(daily_watch_sec) OVER (
        PARTITION BY creator_id 
        ORDER BY ds 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg_sec,
    SUM(daily_watch_sec) OVER (
        PARTITION BY creator_id 
        ORDER BY ds 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_sum_sec
FROM creator_daily_stats;
        """, language="sql")

    else:
        st.markdown("""
        **Problem Statement:** Growth Accounting State Machine. Classify each user into one of four states:
        `New`, `Retained`, `Churned`, or `Resurrected`.
        """)
        st.code("""
SELECT 
    COALESCE(t.user_id, y.user_id) AS user_id,
    '2024-05-14' AS ds,
    CASE 
        WHEN y.user_id IS NULL AND t.user_id IS NOT NULL THEN 'New'
        WHEN y.user_id IS NOT NULL AND t.user_id IS NOT NULL THEN 'Retained'
        WHEN y.user_id IS NOT NULL AND t.user_id IS NULL THEN 'Churned'
        WHEN y.user_id IS NULL AND t.user_id IS NOT NULL AND h.user_id IS NOT NULL THEN 'Resurrected'
    END AS growth_state
FROM today_active_users t
FULL OUTER JOIN yesterday_active_users y ON t.user_id = y.user_id
LEFT JOIN historical_users h ON t.user_id = h.user_id;
        """, language="sql")

    st.markdown("---")
    st.markdown("### Distributed Query Optimization & Skew Mitigation")
    col_skew1, col_skew2 = st.columns([6, 6])
    
    with col_skew1:
        st.markdown("**The Data Skew Problem:**")
        st.write("When joining on a key with extreme power-law distribution (e.g. celebrity account `creator_id = 999999` with 100M rows), Presto/Spark hashes all matching rows to a single reducer node, causing an OOM (Out of Memory) crash or 99% task stall.")
        
        st.markdown("**The Salting Solution:**")
        st.code("""
-- 1. Salt the large skewed fact table
SELECT 
    CONCAT(creator_id, '_', CAST(FLOOR(RAND() * 10) AS VARCHAR)) AS salted_key,
    watch_sec
FROM skewed_fact_impressions;

-- 2. Explode the small dimension table by 10x
SELECT 
    CONCAT(creator_id, '_', CAST(s.salt AS VARCHAR)) AS salted_key,
    creator_name
FROM dim_creator
CROSS JOIN (SELECT explode(sequence(0, 9)) AS salt) s;
        """, language="sql")

    with col_skew2:
        st.markdown("**Distributed Join Strategy Matrix:**")
        st.markdown("""
        | Join Type | Left Table | Right Table | Network Cost | When to Use |
        | :--- | :--- | :--- | :--- | :--- |
        | **Broadcast Join** | Huge (>10TB) | Tiny (<100MB) | Low | Dim table fits in memory on all executors |
        | **Shuffle Hash Join** | Large | Medium | High | Keys hashed and re-partitioned across cluster |
        | **Sort-Merge Join** | Huge | Huge | High | Both datasets pre-sorted and bucketed on join key |
        """)

# ---------------------------------------------------------
# MODULE 4: PYTHON SCRIPTING
# ---------------------------------------------------------
def render_module_4():
    st.title("🐍 Module 4: Python & Data Pipeline Scripting")
    st.caption("Meta DE Coding Interview Focus: Array Algorithms, Streaming Deques & JSON Parsers")
    
    selected_code = st.selectbox(
        "Choose Python Interview Problem:",
        [
            "1. Merge Overlapping Intervals (Total Active Video Time)",
            "2. Top K Frequent Elements / Heavy Hitters (Min-Heap)",
            "3. Flatten Deeply Nested Telemetry JSON",
            "4. Moving Average from Streaming Telemetry Window"
        ]
    )
    
    if "Merge Overlapping" in selected_code:
        st.markdown("### Merge Intervals (`calculate_total_active_time`)")
        st.write("Given an array of video playback time intervals `[start, end]`, merge all overlapping intervals and return the total net active watch time.")
        
        raw_input = st.text_input("Test Interval Array (JSON):", "[[1, 4], [3, 6], [8, 10], [9, 12], [15, 18]]")
        
        try:
            intervals = json.loads(raw_input)
            intervals.sort(key=lambda x: x[0])
            
            merged = []
            for inv in intervals:
                if not merged or merged[-1][1] < inv[0]:
                    merged.append(inv)
                else:
                    merged[-1][1] = max(merged[-1][1], inv[1])
                    
            total_active = sum(inv[1] - inv[0] for inv in merged)
            
            st.success(f"✅ Merged Intervals: `{merged}` | Net Active Duration: **{total_active} seconds**")
            
            fig = go.Figure()
            for idx, inv in enumerate(intervals):
                fig.add_trace(go.Scatter(
                    x=[inv[0], inv[1]], y=[idx, idx],
                    mode='lines+markers', name=f"Raw [{inv[0]},{inv[1]}]",
                    line=dict(width=6)
                ))
            for idx, inv in enumerate(merged):
                fig.add_trace(go.Scatter(
                    x=[inv[0], inv[1]], y=[-1 - idx, -1 - idx],
                    mode='lines+markers', name=f"Merged [{inv[0]},{inv[1]}]",
                    line=dict(color=META_BLUE, width=8)
                ))
            fig.update_layout(
                title="Interval Timeline: Raw vs Merged",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT_LIGHT),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Invalid format: {e}. Provide valid JSON like [[1, 4], [3, 6]]")

        st.code("""
def calculate_total_active_time(intervals: list[list[int]]) -> int:
    # Time Complexity: O(N log N) due to sorting | Space: O(N)
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_end = merged[-1][1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)
        """, language="python")

    elif "Top K" in selected_code:
        st.markdown("### Top K Frequent Elements (`find_heavy_hitters`)")
        st.write("Find the top K most frequent hashtags in a live event stream using a Min-Heap.")
        
        tags_input = st.text_input("Stream of Hashtags (comma separated):", "reels, ai, meta, reels, tech, ai, reels, data, meta, reels")
        k_val = st.slider("Select K Value:", 1, 5, 2)
        
        import collections, heapq
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]
        counts = collections.Counter(tags)
        top_k = heapq.nlargest(k_val, counts.items(), key=lambda item: item[1])
        
        st.success(f"Top {k_val} Heavy Hitters: {top_k}")
        
        st.code("""
import collections, heapq

def top_k_frequent(items: list[str], k: int) -> list[tuple[str, int]]:
    # Time Complexity: O(N + M log K) where M is unique items
    counts = collections.Counter(items)
    # Min-heap of size K keeps memory footprint minimal
    return heapq.nlargest(k, counts.items(), key=lambda x: x[1])
        """, language="python")

    elif "Flatten" in selected_code:
        st.markdown("### Flatten Deeply Nested Telemetry JSON")
        st.write("Convert arbitrary hierarchical telemetry dictionaries into flat single-level schemas suitable for relational data lakes.")
        
        nested_sample = {
            "event_id": "evt_9918",
            "device": {"os": "iOS", "version": {"major": 17, "minor": 4}},
            "geo": {"country": "US", "city": "Seattle"},
            "metrics": {"latency_ms": 142}
        }
        
        st.write("**Input Nested Object:**")
        st.json(nested_sample)
        
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
            
        flat = flatten_dict(nested_sample)
        st.write("**Flattened Lake Record:**")
        st.json(flat)

    else:
        st.markdown("### Moving Average from Data Stream")
        st.code("""
from collections import deque

class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        self.total_sum = 0.0

    def next(self, val: float) -> float:
        # O(1) Time Complexity per event
        self.queue.append(val)
        self.total_sum += val
        if len(self.queue) > self.size:
            self.total_sum -= self.queue.popleft()
        return round(self.total_sum / len(self.queue), 2)
        """, language="python")

# ---------------------------------------------------------
# MODULE 5: BIG DATA PIPELINES & RELIABILITY
# ---------------------------------------------------------
def render_module_5():
    st.title("🔄 Module 5: Big Data Pipelines & Reliability")
    st.caption("Designing Resilient, Idempotent Data Architectures with Circuit Breakers")
    
    st.markdown("### Interactive Idempotent Pipeline Runner")
    st.write("Simulate an enterprise Airflow / Flink DAG with atomic partition swaps and data quality gates.")
    
    col_run1, col_run2 = st.columns(2)
    with col_run1:
        simulate_dq_failure = st.checkbox("🚨 Inject Synthetic Data Quality Anomaly (Null Spike / Duplicate PKs)")
    with col_run2:
        run_btn = st.button("🚀 Trigger Scheduled DAG Execution (ds='2024-05-14')", type="primary")

    if run_btn:
        st.markdown("#### DAG Execution Timeline")
        progress = st.progress(0)
        
        stages = [
            ("Stage 1: Extract & Write to Staging Path", 0.25, "s3://lake/staging/reels/ds=2024-05-14/"),
            ("Stage 2: Run Deduplication & Primary Key Integrity", 0.50, "SELECT user_id, COUNT(*) HAVING COUNT(*) > 1"),
            ("Stage 3: Data Quality Circuit Breaker Evaluation", 0.75, "Verify null_rate < 0.01% & row_count > 10,000,000"),
            ("Stage 4: Atomic Partition Swap to Production Table", 1.0, "ALTER TABLE prod_reels SWAP PARTITION (ds='2024-05-14')")
        ]
        
        for name, pct, detail in stages:
            if simulate_dq_failure and pct == 0.75:
                progress.progress(0.70)
                st.error("❌ CIRCUIT BREAKER TRIGGERED: Quality check failed! 2.8% null user_id detected (threshold = 0.01%).")
                st.warning("⚠️ ROLLBACK EXECUTED: Staging partition discarded. Production partition preserved. PagerDuty alerted.")
                break
            else:
                progress.progress(pct)
                st.success(f"✅ {name}\n\n`{detail}`")
                
        if not simulate_dq_failure:
            st.balloons()

    st.markdown("---")
    st.markdown("### System Architecture Trade-off Matrix")
    req_latency = st.select_slider(
        "Select Business SLA Latency Requirement:",
        options=["Sub-second (<100ms)", "Near Real-Time (1-5 min)", "Hourly (1-2 hr)", "Daily Batch (24 hr)"],
        value="Hourly (1-2 hr)"
    )
    
    if req_latency == "Sub-second (<100ms)":
        st.info("**Architecture: Pure Streaming (Apache Flink + Kafka + RocksDB)**\n\n- Cost: High\n- Complexity: High\n- Storage: In-memory state with checkpointing to S3/HDFS\n- Use Case: Live Ad Bidding, Fraud Detection")
    elif req_latency == "Near Real-Time (1-5 min)":
        st.info("**Architecture: Micro-batch (Spark Structured Streaming + Iceberg)**\n\n- Cost: Medium-High\n- Complexity: Medium\n- Storage: Parquet files with fast metadata commits\n- Use Case: Creator Analytics Dashboard")
    elif req_latency == "Hourly (1-2 hr)":
        st.info("**Architecture: Incremental Batch (Presto/Spark on Hive)**\n\n- Cost: Medium\n- Complexity: Low-Medium\n- Storage: Hourly partition directories (`ds=.../hr=...`)\n- Use Case: Executive Metrics, Campaign Attributions")
    else:
        st.info("**Architecture: Daily Batch (Apache Airflow + Spark Full Overwrite)**\n\n- Cost: Lowest\n- Complexity: Lowest\n- Storage: Daily partitions with atomic swaps\n- Use Case: Financial Auditing, ML Training Set Generation")

# ---------------------------------------------------------
# MODULE 6: AI INFRASTRUCTURE
# ---------------------------------------------------------
def render_module_6():
    st.title("🤖 Module 6: AI Infrastructure & Modern DE Workflows")
    st.caption("Feature Stores, Streaming Vectors (FAISS) & Training-Serving Skew Detection")
    
    st.markdown("### Meta Recommendation Engine Data Flow")
    st.graphviz_chart('''
        digraph G {
            rankdir=LR;
            bgcolor="transparent";
            node [fontname="sans-serif", fontcolor="white", color="#0666FE", style="filled", fillcolor="#1E293B", shape=box, rx=4];
            edge [color="#64748B", fontcolor="#94A3B8", fontname="sans-serif", fontsize=10];
            
            App [label="Client App\\n(Instagram/FB)"];
            Kafka [label="Kafka / Scribe\\n(Event Ingestion)"];
            Flink [label="Apache Flink\\n(Feature Extraction)"];
            Redis [label="Online Feature Store\\n(RocksDB/Redis)"];
            Lake [label="Data Lake\\n(Apache Iceberg)"];
            Spark [label="Spark Offline Job\\n(Batch Aggregates)"];
            FAISS [label="Vector DB\\n(FAISS Embeddings)"];
            Model [label="Ranking Engine\\n(DLRM Model)"];
            
            App -> Kafka [label="Watch / Click"];
            Kafka -> Flink [label="Stream"];
            Flink -> Redis [label="Low-latency write (<10ms)"];
            Kafka -> Lake [label="Raw Parquet"];
            Lake -> Spark [label="Daily Historical"];
            Spark -> Redis [label="Bulk Sync"];
            Spark -> FAISS [label="Index Embeddings"];
            Redis -> Model [label="Realtime Features"];
            FAISS -> Model [label="Candidate Retrieval"];
        }
    ''')
    
    st.markdown("---")
    st.markdown("### Interactive Training-Serving Skew & Drift Monitor")
    st.write("Adjust the serving feature drift to calculate real-time Population Stability Index (PSI).")
    
    drift_factor = st.slider("Simulate Serving Distribution Shift (Mean Offset):", 0.0, 15.0, 4.0, 0.5)
    
    np.random.seed(42)
    train_dist = np.random.normal(50, 10, 2000)
    serve_dist = np.random.normal(50 + drift_factor, 10 + (drift_factor * 0.2), 2000)
    
    # Calculate approximate PSI
    bins = np.linspace(20, 90, 10)
    train_counts, _ = np.histogram(train_dist, bins=bins)
    serve_counts, _ = np.histogram(serve_dist, bins=bins)
    
    train_pct = np.maximum(train_counts / len(train_dist), 0.0001)
    serve_pct = np.maximum(serve_counts / len(serve_dist), 0.0001)
    psi = np.sum((serve_pct - train_pct) * np.log(serve_pct / train_pct))
    
    if psi < 0.1:
        st.success(f"✅ PSI = **{psi:.3f}** (Negligible Drift. Model inference is statistically consistent with training data.)")
    elif psi < 0.25:
        st.warning(f"⚠️ PSI = **{psi:.3f}** (Moderate Feature Drift detected. Queue automated retraining pipeline.)")
    else:
        st.error(f"🚨 PSI = **{psi:.3f}** (CRITICAL TRAINING-SERVING SKEW! Model recommendations are degraded. Fallback to heuristic ranker.)")
        
    fig_drift = go.Figure()
    fig_drift.add_trace(go.Histogram(x=train_dist, name="Training Distribution", opacity=0.6, marker_color=META_BLUE))
    fig_drift.add_trace(go.Histogram(x=serve_dist, name="Live Serving Distribution", opacity=0.6, marker_color="#EF4444"))
    fig_drift.update_layout(
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_LIGHT),
        height=320,
        margin=dict(l=10, r=10, t=20, b=20)
    )
    st.plotly_chart(fig_drift, use_container_width=True)

# ---------------------------------------------------------
# MODULE 7: DATA VISUALIZATION & REPORTING
# ---------------------------------------------------------
def render_module_7():
    st.title("📈 Module 7: Data Visualization & Reporting Strategy")
    st.caption("Cohort Retention Triangles, Funnel Conversions & Executive Telemetry")
    
    st.markdown("### Interactive Funnel Drop-off Analysis")
    
    col_sl1, col_sl2, col_sl3 = st.columns(3)
    with col_sl1:
        watched_3s_conv = st.slider("3s Watch Conversion (%)", 10, 80, 45)
    with col_sl2:
        liked_conv = st.slider("Like Conversion (%)", 1, 40, 12)
    with col_sl3:
        shared_conv = st.slider("Share Conversion (%)", 1, 20, 4)
        
    base_impressions = 1000000
    w3s = base_impressions * (watched_3s_conv / 100)
    liked = w3s * (liked_conv / 100)
    shared = liked * (shared_conv / 100)
    
    fig_funnel = go.Figure(go.Funnel(
        y=["1. Impressions", "2. Watched >= 3s", "3. Liked", "4. Shared"],
        x=[base_impressions, w3s, liked, shared],
        textinfo="value+percent initial+percent previous",
        marker=dict(color=[META_BLUE, "#38BDF8", "#818CF8", "#C084FC"])
    ))
    fig_funnel.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT), height=300)
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("---")
    st.markdown("### Real-world Cohort Retention Triangle")
    st.write("Notice how classic retention triangles leave future dates empty, rather than displaying zeroes.")
    
    cohort_decay = st.slider("Cohort Decay Factor:", 0.4, 0.9, 0.72)
    
    triangle_data = [
        [100.0, round(100.0 * cohort_decay, 1), round(100.0 * (cohort_decay**2), 1), round(100.0 * (cohort_decay**3), 1)],
        [100.0, round(100.0 * cohort_decay, 1), round(100.0 * (cohort_decay**2), 1), None],
        [100.0, round(100.0 * cohort_decay, 1), None, None],
        [100.0, None, None, None]
    ]
    
    fig_heat = px.imshow(
        triangle_data,
        x=["Day 0", "Day 1", "Day 7", "Day 30"],
        y=["2024-W16", "2024-W17", "2024-W18", "2024-W19"],
        color_continuous_scale="Blues",
        labels=dict(x="Retention Window", y="Acquisition Cohort", color="Retention %"),
        text_auto=True
    )
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT), height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------
# MODULE 8: BEHAVIORAL & OWNERSHIP
# ---------------------------------------------------------
def render_module_8():
    st.title("🎯 Module 8: Behavioral & Ownership (Meta Core Values)")
    st.caption("How to Structure DE Leadership, Trade-offs & Conflict Stories via the STAR Method")
    
    st.markdown("""
    <div class="card">
        <div class="label">Meta's 6 Official Core Values (Updated)</div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; font-size: 12px;">
            <div><strong>1. Move Fast</strong><br/><span style="color:#94A3B8; font-size: 11px;">Iterate rapidly without fear of breaking things responsibly.</span></div>
            <div><strong>2. Focus on Long-Term Impact</strong><br/><span style="color:#94A3B8; font-size: 11px;">Solve foundational engineering problems, not just symptoms.</span></div>
            <div><strong>3. Build Awesome Things</strong><br/><span style="color:#94A3B8; font-size: 11px;">Raise the bar for quality and user craft.</span></div>
            <div><strong>4. Live in the Future</strong><br/><span style="color:#94A3B8; font-size: 11px;">Embrace decentralized systems, AI workflows, and VR/MR.</span></div>
            <div><strong>5. Be Direct & Respect Colleagues</strong><br/><span style="color:#94A3B8; font-size: 11px;">Give clear, actionable feedback and debate rigorously.</span></div>
            <div><strong>6. Meta, Metamates, Me</strong><br/><span style="color:#94A3B8; font-size: 11px;">Stewardship for the company mission over individual siloed credit.</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Interactive STAR Story Builder")
    selected_val = st.selectbox(
        "Target Meta Core Value:",
        ["Move Fast", "Focus on Long-Term Impact", "Build Awesome Things", "Live in the Future", "Be Direct and Respect Your Colleagues", "Meta, Metamates, Me"]
    )
    
    story_title = st.text_input("Story Title:", "Overcame 3-Hour Pipeline SLA Breach by Refactoring Shuffle Partitioning")
    
    s_col, t_col = st.columns(2)
    with s_col:
        sit = st.text_area("S - Situation (Context, scale, problem constraint):", "During the Black Friday traffic surge, our main attribution pipeline SLA breached by 3.5 hours, delaying advertiser billing metrics.")
    with t_col:
        tsk = st.text_area("T - Task (Your individual responsibility & ownership):", "As the lead DE on core monetization, I was tasked with diagnosing the bottleneck and restoring the SLA under 45 minutes without tripling cluster cost.")
        
    a_col, r_col = st.columns(2)
    with a_col:
        act = st.text_area("A - Action (Technical actions, trade-offs, courage):", "Profiled Spark DAG executors and identified extreme key skew in our ad_id join. Implemented salting on the top 100 viral campaigns and converted the advertiser dimension to a broadcast join.")
    with r_col:
        res = st.text_area("R - Result (Quantifiable business & engineering metrics):", "Pipeline runtime dropped from 4.5 hours to 38 minutes (86% reduction), saving $12,000/month in cloud compute and achieving 100% SLA compliance.")

    if st.button("💾 Save STAR Story to My Blueprint Bank", type="primary"):
        st.session_state.star_stories.append({
            "value": selected_val,
            "title": story_title,
            "summary": f"**Situation:** {sit}\n\n**Action:** {act}\n\n**Result:** {res}"
        })
        st.success("Story successfully saved!")

    if st.session_state.star_stories:
        st.markdown("#### 📖 Saved STAR Stories:")
        for idx, item in enumerate(st.session_state.star_stories):
            with st.expander(f"⭐ [{item['value']}] {item['title']}"):
                st.markdown(item["summary"])

    st.markdown("---")
    st.markdown("### ⏱️ Timed Mock Technical Screen Simulator")
    st.write("Meta technical phone screens allow **exactly 45 minutes** for 1 SQL and 1 Python coding problem.")
    
    if st.button("Start 45-Minute Practice Clock"):
        st.info("⏰ Timer active! Target pacing: 20 min for SQL (Window functions), 20 min for Python (Algorithms), 5 min for questions.")

# ---------------------------------------------------------
# MODULE 9: 4-WEEK ACTION ROADMAP
# ---------------------------------------------------------
def render_module_9():
    st.title("📅 Module 9: Actionable 4-Week Preparation Roadmap")
    st.caption("Day-by-Day Milestone Plan & Progress Tracker")
    
    tasks = st.session_state.checklist_state
    completed = sum(1 for v in tasks.values() if v)
    total = len(tasks)
    progress_ratio = completed / total if total > 0 else 0
    
    st.progress(progress_ratio)
    st.markdown(f"**Readiness Score:** `{completed} / {total}` Core Competencies Completed (`{int(progress_ratio*100)}%`)")
    
    st.write("---")
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("#### Week 1: SQL Mastery & Array Algorithms")
        for t in ["SQL Window Functions (Dense_Rank, Row_Number)", "Python Interval Merging & Stream Deque", "Star Schema vs Snowflake Trade-offs"]:
            tasks[t] = st.checkbox(t, value=tasks.get(t, False), key=f"chk_{t}")
            
        st.markdown("#### Week 2: Advanced Data Modeling & Dimensions")
        for t in ["SCD Type 2 Dimension Implementation", "Presto / Spark Shuffle & Salting for Data Skew", "Broadcast vs Hash Joins & Partition Pruning"]:
            tasks[t] = st.checkbox(t, value=tasks.get(t, False), key=f"chk_{t}")

    with col_w2:
        st.markdown("#### Week 3: Big Data Pipelines & Metric RCA")
        for t in ["Idempotent Backfill & Circuit Breakers", "Product Sense: Root Cause Analysis (RCA) Framework", "Feature Store & Training-Serving Skew Monitoring"]:
            tasks[t] = st.checkbox(t, value=tasks.get(t, False), key=f"chk_{t}")

        st.markdown("#### Week 4: Behavioral & Final Mock Screens")
        for t in ["Cohort Retention Triangles & Funnel Conversions", "Write 5 Meta Core Values STAR Stories", "Complete 2 Full 45-min Timed Mock Screens"]:
            tasks[t] = st.checkbox(t, value=tasks.get(t, False), key=f"chk_{t}")

    st.markdown("---")
    st.download_button(
        label="📥 Export Preparation Progress (JSON)",
        data=json.dumps(tasks, indent=2),
        file_name="meta_de_prep_checklist.json",
        mime="application/json"
    )

# ---------------------------------------------------------
# MODULE 10: META TECH RADAR & ENGINEERING BLOGS
# ---------------------------------------------------------
def render_module_10():
    st.title("📡 Module 10: Meta Tech Radar & Engineering Publications")
    st.caption("Current Data Engineering Trends & Meta-Specific Engineering Blogs (engineering.fb.com) with Google Search Grounding")
    
    st.markdown("""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="label" style="color: #60A5FA;">Real-Time Architectural Intelligence</div>
                <div style="font-size: 13px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">
                    Google Search Grounded Semantic Knowledge Base
                </div>
            </div>
            <button onclick="if(window.parent && window.parent.toggleSearchSidebar){window.parent.toggleSearchSidebar(true)}else if(window.toggleSearchSidebar){window.toggleSearchSidebar(true)}" style="background: #0666FE; color: white; border: none; border-radius: 4px; padding: 6px 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; cursor: pointer;">
                Open Slide-Over Radar 🔍
            </button>
        </div>
        <p style="font-size: 11.5px; color: #94A3B8; margin-top: 8px; margin-bottom: 0;">
            Meta's data infrastructure evolves rapidly. Interviewers expect candidates to discuss modern paradigms beyond legacy Hadoop/Hive: C++ vectorized query execution (<a href="https://engineering.fb.com" target="_blank" style="color: #60A5FA;">Velox</a>), table formats (<a href="https://iceberg.apache.org" target="_blank" style="color: #60A5FA;">Apache Iceberg</a>), stateful stream processing (<a href="https://flink.apache.org" target="_blank" style="color: #60A5FA;">Flink</a>), and in-memory telemetry (<a href="https://engineering.fb.com" target="_blank" style="color: #60A5FA;">Scuba/Gorilla</a>).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Interactive In-App Semantic Search")
    
    col_q1, col_q2 = st.columns([8, 2])
    with col_q1:
        query_input = st.text_input(
            "Search Query:",
            value="Meta Velox unified execution engine Presto Spark",
            placeholder="e.g. Meta Iceberg adoption, Flink Reels streaming, FAISS vector search..."
        )
    with col_q2:
        st.write("")
        st.write("")
        trigger_search = st.button("Search Grounding", type="primary")

    if trigger_search and query_input:
        with st.spinner("Querying Google Search Grounding & Synthesizing Meta Engineering Publications..."):
            try:
                import urllib.parse
                import json
                encoded_q = urllib.parse.quote(query_input.strip())
                
                # Fetch through Pyodide open_url or standard urllib
                try:
                    from pyodide.http import open_url
                    res = open_url(f"/api/search?q={encoded_q}")
                    payload = json.loads(res.read())
                except Exception:
                    import urllib.request
                    with urllib.request.urlopen(f"http://localhost:3000/api/search?q={encoded_q}", timeout=30) as resp:
                        payload = json.loads(resp.read().decode('utf-8'))

                if "error" in payload:
                    st.error(f"Search Error: {payload['error']}")
                else:
                    st.markdown("#### Grounded Architecture Brief")
                    st.markdown(payload.get("content", ""))
                    
                    sources = payload.get("sources", [])
                    if sources:
                        st.markdown("---")
                        st.markdown("**🔗 Verified Grounded Sources & Publications:**")
                        source_cols = st.columns(min(len(sources), 3))
                        for idx, src in enumerate(sources):
                            with source_cols[idx % 3]:
                                st.markdown(f"- [{src.get('title', 'Source')}]({src.get('uri', '#')})")

            except Exception as e:
                st.warning(f"Note: Connected through local backend. For quick search with real-time sliding panel, click the 'Meta Tech Radar' button in the top right corner. (Error: {e})")

    st.markdown("---")
    st.markdown("### 📚 Meta Engineering Landmark Papers & Systems Guide")
    st.caption("Essential systems referenced in Meta DE Case Study & System Design rounds")

    tech_items = [
        {
            "name": "Velox: Unified Execution Engine",
            "year": "2022-Present",
            "category": "Query Engine & Vectorization",
            "summary": "Meta's open-source C++ database acceleration library. Eliminates redundant query engines by providing a single vectorized execution runtime for Presto, Apache Spark, and PyTorch dataloaders.",
            "link": "https://engineering.fb.com/2023/03/09/open-source/velox-open-source-execution-engine/",
            "interview_tip": "Highlight how vectorized execution (SIMD) and column-batch processing in C++ bypass JVM garbage collection and JIT overhead at exabyte scale."
        },
        {
            "name": "Apache Iceberg at Meta Scale",
            "year": "2023-Present",
            "category": "Data Lakehouse & Table Formats",
            "summary": "Meta's migration from legacy Hive metastore partition directories to Apache Iceberg metadata tree structures, enabling ACID transactions, atomic partition swaps, and hidden partitioning.",
            "link": "https://engineering.fb.com/2024/01/23/data-infrastructure/apache-iceberg-meta/",
            "interview_tip": "Discuss how Iceberg eliminates the O(N) filesystem list directory bottleneck during query planning by tracking files directly in snapshot metadata manifests."
        },
        {
            "name": "Apache Flink Real-Time Streaming",
            "year": "Core Infrastructure",
            "category": "Stream Processing",
            "summary": "Powers Instagram Reels real-time ranking, recommendation signal extraction, and ad attribution with sub-second SLA and RocksDB state backend.",
            "link": "https://engineering.fb.com/2021/04/12/developer-tools/real-time-analytics/",
            "interview_tip": "Explain watermarking, exactly-once processing with two-phase commit sinks, and managing large RocksDB state checkpoints."
        },
        {
            "name": "Scuba & Gorilla Telemetry",
            "year": "Landmark Systems",
            "category": "Real-Time In-Memory Analytics",
            "summary": "Scuba provides real-time distributed log analytics (<1 second response on billions of events), while Gorilla provides high-density in-memory time-series storage with XOR double-delta floating point compression.",
            "link": "https://engineering.fb.com/2015/08/27/core-infra/gorilla-a-fast-scalable-in-memory-time-series-database/",
            "interview_tip": "Use Gorilla as an example of domain-specific lossless compression: timestamps compress by 96% and float values by 73% using XOR delta encoding."
        },
        {
            "name": "FAISS & DLRM AI Infrastructure",
            "year": "AI / Recommendations",
            "category": "Vector Similarity & Ranking",
            "summary": "FAISS (Facebook AI Similarity Search) enables billion-scale approximate nearest neighbor (ANN) retrieval for dense embeddings, feeding candidate sets into Deep Learning Recommendation Models.",
            "link": "https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/",
            "interview_tip": "Clarify the two-stage recommendation pipeline: Stage 1 (FAISS Candidate Generation: millions to thousands) and Stage 2 (DLRM Heavy Ranking: thousands to top 10 items)."
        },
        {
            "name": "Data Quality Circuit Breakers",
            "year": "Pipeline Reliability",
            "category": "Governance & Integrity",
            "summary": "Automated data verification gates embedded between staging partitions and production tables. Detects null spikes, duplicate keys, and volume anomalies, preventing corrupt data from polluting downstream dashboards.",
            "link": "https://engineering.fb.com",
            "interview_tip": "Stress the importance of automated circuit breakers: it is always better to delay an SLA by 30 minutes than to publish corrupted financial or metric data."
        }
    ]

    for item in tech_items:
        with st.expander(f"🔹 {item['name']} ({item['category']} • {item['year']})"):
            st.markdown(f"**Architecture Summary:** {item['summary']}")
            st.markdown(f"**🎯 Interview Talking Point:** {item['interview_tip']}")
            st.markdown(f"🔗 [Read Official Meta Engineering Post]({item['link']})")

# ---------------------------------------------------------
# MAIN NAVIGATION ROUTER
# ---------------------------------------------------------
pages = {
    "Overview & Dashboard": render_dashboard,
    "Module 1: Product Sense & Metrics": render_module_1,
    "Module 2: Data Modeling & Schema Design": render_module_2,
    "Module 3: SQL Engine & Optimization": render_module_3,
    "Module 4: Python Scripting": render_module_4,
    "Module 5: Big Data Pipelines": render_module_5,
    "Module 6: AI Infrastructure": render_module_6,
    "Module 7: Data Visualization": render_module_7,
    "Module 8: Behavioral (Meta Values)": render_module_8,
    "Module 9: 4-Week Roadmap": render_module_9,
    "Module 10: Meta Tech Radar & Blogs": render_module_10
}

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; padding: 4px;">
    <div style="width: 28px; height: 28px; background: {META_BLUE}; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: white;">M</div>
    <div>
        <div style="font-weight: 700; font-size: 13px; color: #FFFFFF; letter-spacing: -0.02em;">META DE BLUEPRINT</div>
        <div style="font-size: 10px; color: {TEXT_MUTED}; text-transform: uppercase;">L4/L5 Interview Guide</div>
    </div>
</div>
""", unsafe_allow_html=True)

selection = st.sidebar.radio("Preparation Modules:", list(pages.keys()))

st.sidebar.markdown("---")
total_t = len(st.session_state.checklist_state)
done_t = sum(1 for v in st.session_state.checklist_state.values() if v)
readiness = int((done_t / total_t) * 100) if total_t else 0

st.sidebar.markdown(f"""
<div style="margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
    <span class="label">Candidate Readiness</span>
    <span style="font-size: 11px; font-weight: 700; color: {META_BLUE}; font-family: monospace;">{readiness}%</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.progress(readiness / 100)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background: #162032; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: #60A5FA; letter-spacing: 0.05em; margin-bottom: 4px;">📡 Meta Tech Radar</div>
    <div style="font-size: 11px; color: #94A3B8; margin-bottom: 8px;">Semantic Google Search Grounding for current DE trends & Meta engineering blogs.</div>
    <button onclick="if(window.parent && window.parent.toggleSearchSidebar){window.parent.toggleSearchSidebar(true)}else if(window.toggleSearchSidebar){window.toggleSearchSidebar(true)}" style="width: 100%; background: {META_BLUE}; color: white; border: none; border-radius: 4px; padding: 6px 10px; font-size: 10.5px; font-weight: 700; text-transform: uppercase; cursor: pointer;">Open Radar Search 🔍</button>
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("Designed for Meta DE candidates • Powered by Pyodide, Express & Gemini 3.8")

# Route to chosen page
pages[selection]()

