
import os, json, pathlib
import gradio as gr
from simulation_core import SimulationEngine, load_personas, Persona, save_run_md, save_run_json

APP_DIR = pathlib.Path(__file__).parent.resolve()
personas = load_personas(str(APP_DIR / "personas.yaml"))
engine = SimulationEngine()

def list_personas():
    return {f"{p.name} — {p.archetype} ({pid})": pid for pid,p in personas.items()}

def run(feature_title, spec, persona_ids, turns, seed):
    records = []
    for pid in persona_ids:
        conv = engine.simulate(personas[pid], feature_title, spec, turns=int(turns), seed=int(seed))
        md = save_run_md(conv, str(APP_DIR))
        js = save_run_json(conv, str(APP_DIR))
        chat = ""
        for m in conv.messages:
            if m.role == "user":
                chat += f"<div style='margin:6px 0'><b>User:</b> {m.content}</div>"
            else:
                chat += f"<div style='margin:6px 0'><b>{conv.persona.name}:</b> {m.content}</div>"
        records.append((conv.persona.name, conv.feature_title, chat, md, js))
    return records

with gr.Blocks(title="Persona Conversation Simulator") as demo:
    gr.Markdown("# 🎭 Persona Conversation Simulator (Gradio)")
    with gr.Row():
        feature_title = gr.Textbox(label="Feature Title", placeholder="Smart Collections (AI auto‑tagging)")
        turns = gr.Slider(2, 8, value=4, step=1, label="Turns")
    spec = gr.Textbox(lines=10, label="Feature Description & Context")
    with gr.Row():
        persona_select = gr.CheckboxGroup(choices=list(list_personas().keys()), label="Personas")
        seed = gr.Number(value=42, label="Random Seed")
    run_btn = gr.Button("Run Simulation")
    out = gr.Dataset(components=[gr.Textbox(label="Persona"),
                                 gr.Textbox(label="Feature"),
                                 gr.HTML(label="Chat"),
                                 gr.Textbox(label="Markdown Path"),
                                 gr.Textbox(label="JSON Path")],
                     label="Results (click rows to copy fields)")
    run_btn.click(
        lambda ft, sp, chs, t, sd: run(ft, sp, [list_personas()[c] for c in (chs or [])], t, sd),
        [feature_title, spec, persona_select, turns, seed],
        out
    )

if __name__ == "__main__":
    demo.launch()
