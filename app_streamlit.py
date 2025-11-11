
import os, json, textwrap, pathlib
import streamlit as st
import pandas as pd
from simulation_core import SimulationEngine, load_personas, Persona, save_run_md, save_run_json

APP_DIR = pathlib.Path(__file__).parent.resolve()
st.set_page_config(page_title="Persona Conversation Simulator", page_icon="🎭", layout="wide")

st.title("🎭 Persona Conversation Simulator")
st.caption("Simulate user conversations and feedback by persona — compare, save, export.")

with st.sidebar:
    st.header("Settings")
    use_tiny = st.toggle("Use TinyTroupe (if available)", value=True)
    st.session_state['use_tiny'] = use_tiny
    model = st.text_input("Model hint (optional)", placeholder="e.g., gpt-4.1-mini")
    base_dir = st.text_input("Storage directory", value=str(APP_DIR))
    st.info("Tip: Without API keys or TinyTroupe, the app uses a heuristic fallback that requires no internet.", icon="💡")

personas = load_personas(str(APP_DIR / "personas.yaml"))
persona_ids = list(personas.keys())

st.subheader("1) Feature Specification")
col1, col2 = st.columns([1,1])
with col1:
    feature_title = st.text_input("Feature title", placeholder="e.g., Smart Collections (AI auto‑tagging)")
with col2:
    batch_variants = st.text_area("Batch variants (optional)", placeholder="One title per line to run in batch — uses same spec.")

spec = st.text_area("Feature description & context", height=220, placeholder=textwrap.dedent("""
    Include:
    - Primary job-to-be-done
    - User flow (steps), key visuals, states
    - Metrics/guardrails (e.g., activation rate, D1 retention target)
    - Constraints (e.g., mobile-first, accessibility, compliance)
""").strip())

st.subheader("2) Persona Selection")
left, right = st.columns([1,1])
with left:
    selected = st.multiselect("Predefined personas", options=persona_ids, format_func=lambda i: f"{personas[i].name} — {personas[i].archetype}")
with right:
    st.markdown("**Custom persona (optional)**")
    with st.expander("Create custom persona"):
        cust = {}
        cust["id"] = st.text_input("ID", value="custom_1")
        cust["name"] = st.text_input("Name", value="Alex")
        cust["archetype"] = st.text_input("Archetype", value="Time-pressed new user")
        cust["traits"] = st.text_input("Traits (comma-separated)", value="curious, impatient, visual-first")
        cust["goals"] = st.text_input("Goals (comma-separated)", value="finish setup, understand value")
        cust["tech_literacy"] = st.selectbox("Tech literacy", ["novice","intermediate","advanced","expert"], index=1)
        cust["risk_aversion"] = st.selectbox("Risk aversion", ["low","medium","high"], index=1)
        cust["tone"] = st.selectbox("Tone", ["casual","neutral","formal"], index=1)
        cust["pain_points"] = st.text_input("Pain points", value="too many steps, unclear labels")
        cust["motivations"] = st.text_input("Motivations", value="speed, clarity, social proof")
        cust["domain_expertise"] = st.text_input("Domain expertise", value="onboarding, mobile UX")
        cust["context"] = st.text_area("Context", value="First day using the product; has to invite a teammate.")
        add_custom = st.button("Add custom persona")
        if add_custom:
            p = Persona(
                id=cust["id"],
                name=cust["name"],
                archetype=cust["archetype"],
                traits=[x.strip() for x in cust["traits"].split(",") if x.strip()],
                goals=[x.strip() for x in cust["goals"].split(",") if x.strip()],
                tech_literacy=cust["tech_literacy"],
                risk_aversion=cust["risk_aversion"],
                tone=cust["tone"],
                pain_points=[x.strip() for x in cust["pain_points"].split(",") if x.strip()],
                motivations=[x.strip() for x in cust["motivations"].split(",") if x.strip()],
                domain_expertise=[x.strip() for x in cust["domain_expertise"].split(",") if x.strip()],
                context=cust["context"]
            )
            personas[p.id] = p
            st.success(f"Added custom persona '{p.name}' ({p.id}).")

st.subheader("3) Run Simulation")
turns = st.slider("Number of turns per persona", 2, 8, 4)
seed = st.number_input("Random seed", value=42)
run_btn = st.button("Run (single or batch)", type="primary", disabled=(not selected and "custom_1" not in personas))

variant_titles = [t.strip() for t in (batch_variants or "").split("\\n") if t.strip()]
if feature_title.strip():
    variant_titles = [feature_title.strip()] + variant_titles
variant_titles = list(dict.fromkeys(variant_titles))

if run_btn and variant_titles and spec.strip():
    engine = SimulationEngine(model=model if st.session_state.get("use_tiny", True) else None)
    records = []
    for vt in variant_titles:
        for pid in selected or []:
            conv = engine.simulate(personas[pid], vt, spec, turns=turns, seed=int(seed))
            md_path = save_run_md(conv, base_dir)
            js_path = save_run_json(conv, base_dir)
            records.append(dict(feature_title=vt, persona=personas[pid].name, id=conv.feature_id, md=md_path, json=js_path))

    st.success(f"Completed {len(records)} simulation(s). Saved under: {base_dir}/runs")
    import pandas as pd
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)
    st.download_button("Export table (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="simulations_index.csv", mime="text/csv")

    if records:
        last = records[-1]
        st.subheader("Latest Conversation (chat view)")
        with open(last["json"], "r") as f:
            conv = json.load(f)
        for msg in conv["messages"]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🎭"):
                    st.markdown(msg["content"])
                    meta = msg.get("meta", {})
                    if meta:
                        st.expander("Persona meta").json(meta)

st.divider()
st.subheader("4) Compare Saved Runs")
runs_dir = pathlib.Path(base_dir)/"runs"
if runs_dir.exists():
    files = sorted([p for p in runs_dir.glob("*.json")], key=lambda p: p.stat().st_mtime, reverse=True)
    options = [p.name for p in files]
    picked = st.multiselect("Select runs to compare", options[:50], max_selections=6)
    if picked:
        rows = []
        for name in picked:
            with open(runs_dir/name, "r") as f:
                conv = json.load(f)
            persona = conv["persona"]["name"]
            feature = conv["feature_title"]
            confidences = [m.get("meta",{}).get("confidence") or m.get("meta",{}).get("confidence", 0.5) for m in conv["messages"] if m["role"]=="persona"]
            conf = round(sum([c for c in confidences if isinstance(c,(int,float))])/max(1,len(confidences)), 2) if confidences else None
            rows.append(dict(file=name, feature=feature, persona=persona, avg_confidence=conf))
        comp = pd.DataFrame(rows)
        st.dataframe(comp, use_container_width=True)
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = plt.gca()
            ax.bar(comp["persona"], comp["avg_confidence"])
            ax.set_title("Average Confidence by Persona")
            ax.set_ylim(0,1.0)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Chart error: {e}")
else:
    st.info("No saved runs yet. Run a simulation to populate this section.")
