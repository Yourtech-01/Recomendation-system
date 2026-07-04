"""
ui/app.py
Streamlit demo for the recommendation engine.
Shows live recommendations, item-to-item similarity, and A/B test CTR.

Run: streamlit run ui/app.py
"""

import os, requests, random
import streamlit as st
import plotly.graph_objects as go

# Streamlit Community Cloud secrets (Settings -> Secrets) aren't auto-exported
# to os.environ, so check st.secrets first and fall back to an env var for
# local runs (`API_URL=http://localhost:8000 streamlit run ui/app.py`).
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

st.set_page_config(page_title="Rec Engine Demo", page_icon="", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem}
.item-card{background:var(--color-background-secondary);border-radius:8px;
           padding:10px 14px;margin-bottom:6px;font-size:13px;
           border:0.5px solid var(--color-border-tertiary)}
.score-bar{height:4px;background:var(--color-background-info);border-radius:2px;margin-top:6px}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Controls")
    user_id  = st.slider("User ID", 0, 1999, 42)
    top_k    = st.slider("Top-K recommendations", 5, 20, 10)
    ab_group = st.radio("A/B group", ["treatment (two-tower)", "control (baseline)"])
    group    = "treatment" if "treatment" in ab_group else "control"
    st.markdown("---")
    try:
        m = requests.get(f"{API_URL}/metrics", timeout=2).json()
        st.markdown("### Live metrics")
        st.metric("Total requests", m.get("total_requests", 0))
        st.metric("P99 latency", f"{m.get('p99_latency_ms', 0)} ms")
        st.metric("Index size", f"{m.get('index_size', 0):,} items")
    except:
        st.warning("API not reachable — start with: uvicorn api.main:app --reload")

st.markdown("# Recommendation engine demo")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### Personalised recommendations")
    try:
        resp = requests.post(f"{API_URL}/recommend",
                             json={"user_id": user_id, "top_k": top_k,
                                   "ab_group": group}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.caption(f"Model: `{data['model']}` · Latency: `{data['latency_ms']} ms`")
            max_score = max(data["scores"]) if data["scores"] else 1
            for rank, (item_id, score) in enumerate(
                    zip(data["items"], data["scores"]), 1):
                bar_pct = int(score / max_score * 100)
                if st.button(f"#{rank}  Item {item_id}   score: {score:.3f}",
                             key=f"item_{item_id}",
                             use_container_width=True):
                    requests.post(f"{API_URL}/ab/click",
                                  params={"user_id": user_id,
                                          "item_id": item_id,
                                          "ab_group": group})
                    st.toast(f"Click recorded for item {item_id}")
        else:
            st.error(f"API error: {resp.text}")
    except Exception as e:
        st.error(f"Cannot connect to API: {e}")

with col2:
    st.markdown("### Item similarity")
    query_item = st.number_input("Query item ID", 0, 4999, 100)
    if st.button("Find similar items"):
        try:
            resp = requests.get(f"{API_URL}/similar/{query_item}",
                                params={"top_k": 8}, timeout=5)
            if resp.status_code == 200:
                sim = resp.json()["similar"]
                st.caption(f"Items most similar to item {query_item}")
                for s in sim:
                    st.markdown(
                        f"Item **{s['item_id']}** — cosine score `{s['score']:.4f}`")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("### A/B test results")
try:
    ab = requests.get(f"{API_URL}/ab/stats", timeout=2).json()
    if ab:
        c1, c2 = st.columns(2)
        for col, (group_name, stats) in zip([c1, c2], ab.items()):
            with col:
                st.metric(f"{group_name} CTR",
                          f"{stats.get('ctr', 0):.2f}%",
                          delta=None)
                st.caption(f"Impressions: {stats.get('impressions',0)} · "
                           f"Clicks: {stats.get('clicks',0)}")
    else:
        st.info("No A/B data yet — make some recommendations above and click items.")
except:
    st.info("Start the API to see live A/B stats.")
