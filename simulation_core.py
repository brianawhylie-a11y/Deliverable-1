
from __future__ import annotations
import os, json, random, time, pathlib, datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple

TT_AVAILABLE = False
try:
    import tinytroupe as tt  # type: ignore
    TT_AVAILABLE = True
except Exception:
    TT_AVAILABLE = False

@dataclass
class Persona:
    id: str
    name: str
    archetype: str
    demographics: Dict[str, Any] = field(default_factory=dict)
    traits: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    tech_literacy: str = "intermediate"
    risk_aversion: str = "medium"
    tone: str = "neutral"
    pain_points: List[str] = field(default_factory=list)
    motivations: List[str] = field(default_factory=list)
    domain_expertise: List[str] = field(default_factory=list)
    context: str = ""

@dataclass
class Message:
    role: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conversation:
    feature_id: str
    feature_title: str
    feature_spec: str
    persona: Persona
    seed: int
    messages: List[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat()+"Z")

    def to_markdown(self) -> str:
        header = f"# Conversation — {self.feature_title} — {self.persona.name} ({self.persona.archetype})\n"
        meta = f"- Created: {self.created_at}\n- Seed: {self.seed}\n\n---\n"
        body = ""
        for m in self.messages:
            role = "👤 User" if m.role == "user" else f"🧪 {self.persona.name}"
            body += f"**{role}:** {m.content}\n\n"
            if m.meta:
                body += f"<details><summary>meta</summary>\n\n```json\n{json.dumps(m.meta, indent=2)}\n```\n</details>\n\n"
        return header + meta + body

def _score_confidence(persona: Persona, text: str) -> float:
    base = 0.55
    if persona.tech_literacy == "expert":
        base += 0.15
    if persona.risk_aversion == "high":
        base -= 0.05
    if any(k in text.lower() for k in ["unclear", "confusing", "not sure"]):
        base -= 0.2
    return max(0.05, min(0.95, base + (random.random()-0.5)*0.1))

def _extract_followups(persona: Persona, spec: str) -> List[str]:
    qs = []
    low = spec.lower()
    if "analytics" in low:
        qs.append("Which success metrics and target thresholds are defined?")
    if "onboarding" in low:
        qs.append("Can users skip steps and finish later?")
    if "paywall" in low or "pricing" in low or "trial" in low:
        qs.append("Is there a free trial or grace period before payment is required?")
    if any(k in low for k in ["accessibility","aria","keyboard","contrast","wcag"]):
        qs.append("Have you tested keyboard-only navigation and screen-reader labels?")
    if "privacy" in low or "gdpr" in low:
        qs.append("What data is collected and how is it governed?")
    if not qs:
        qs.append("What is the primary job-to-be-done for this feature?")
    return qs[:6]

def _heuristic_persona_reply(persona: Persona, user_msg: str, feature_spec: str) -> Tuple[str, Dict[str, Any]]:
    lower = (user_msg + " " + feature_spec).lower()
    concerns, kudos, suggestions = [], [], []

    if "onboarding" in lower or "signup" in lower:
        concerns.append("Onboarding may be too long; consider progressive disclosure.")
    if "dashboard" in lower:
        suggestions.append("Surface the top 3 KPIs above the fold; defer advanced charts.")
    if "mobile" in lower:
        concerns.append("Ensure tap targets are >= 44px and content is responsive.")
    if any(k in lower for k in ["access","aria","keyboard","contrast","wcag","color"]):
        suggestions.append("Add aria-labels, visible focus states, and non-color cues to meet WCAG 2.2 AA.")
    if any(k in lower for k in ["paywall","pricing","trial"]):
        concerns.append("Hard paywalls can increase drop-off; test a trial or freemium.")
    if "privacy" in lower or "gdpr" in lower:
        concerns.append("Clarify data processing, retention, and opt-out; add a DPIA summary.")
    if "api" in lower or "export" in lower:
        suggestions.append("Provide API endpoints and bulk export options for power users.")
    if persona.archetype.lower().startswith("security") or "security" in persona.archetype.lower():
        concerns.append("Need SSO/MFA, audit logs, and data retention controls.")
    if "product manager" in persona.archetype.lower():
        suggestions.append("Define success metrics and an A/B test plan.")
    if "screen" in persona.archetype.lower() or "accessibility" in persona.archetype.lower():
        concerns.append("Verify no keyboard traps; test with VoiceOver/NVDA.")
    if "gen z" in persona.archetype.lower() or persona.tone == "casual":
        kudos.append("Looks fun if visuals stay bold with quick motion affordances.")

    confidence = _score_confidence(persona, user_msg)
    followups = _extract_followups(persona, feature_spec)

    summary = []
    if kudos: summary.append("**What I like:** " + "; ".join(kudos))
    if concerns: summary.append("**Concerns:** " + "; ".join(concerns))
    if suggestions: summary.append("**Suggestions:** " + "; ".join(suggestions))
    if not summary:
        summary.append("Looks promising; would like to test a clickable prototype.")

    reasoning = {
        "salient_traits": persona.traits[:3],
        "assumptions": ["User intent inferred from provided spec", "No live usability data"],
        "signals": {"keywords_hit": [k for k in ["onboarding","dashboard","mobile","access","paywall","pricing","aria","privacy","gdpr","keyboard","contrast"] if k in lower]},
        "confidence": round(confidence, 2),
        "follow_ups": followups
    }
    reply = "\n\n".join(summary)
    return reply, reasoning

class SimulationEngine:
    def __init__(self, model: Optional[str]=None):
        self.model = model

    def simulate(self, persona: Persona, feature_title: str, feature_spec: str, system_preamble: Optional[str]=None, turns:int=4, seed:int=42) -> Conversation:
        random.seed(seed)
        conv = Conversation(
            feature_id = f"{int(time.time())}-{random.randint(1000,9999)}",
            feature_title = feature_title.strip() or "Untitled Feature",
            feature_spec = feature_spec.strip(),
            persona = persona,
            seed = seed,
        )
        user_open = f"Please review the feature: {feature_title}. Key context:\n{feature_spec}\nShare candid feedback in your usual tone."
        conv.messages.append(Message(role="user", content=user_open))

        if TT_AVAILABLE and (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")):
            tiny_person = tt.TinyPerson(
                name = persona.name,
                description = f"{persona.archetype}. Traits: {', '.join(persona.traits)}. Goals: {', '.join(persona.goals)}.",
                memory = f"Context: {persona.context}. Pain points: {', '.join(persona.pain_points)}. Domain expertise: {', '.join(persona.domain_expertise)}."
            )
            prompt = user_open
            for t in range(turns):
                response = tiny_person.listen_and_act(prompt)
                meta = {"source":"tinytroupe","turn":t+1}
                conv.messages.append(Message(role="persona", content=str(response), meta=meta))
                follow_qs = _extract_followups(persona, feature_spec)
                prompt = "Follow-up: " + random.choice(follow_qs)
                conv.messages.append(Message(role="user", content=prompt))
        else:
            reply, meta = _heuristic_persona_reply(persona, user_open, feature_spec)
            conv.messages.append(Message(role="persona", content=reply, meta=meta))
            for t in range(1, turns):
                q = random.choice(meta["follow_ups"])
                conv.messages.append(Message(role="user", content=q))
                follow_reply, meta2 = _heuristic_persona_reply(persona, q, feature_spec)
                conv.messages.append(Message(role="persona", content=follow_reply, meta=meta2))
        return conv

def load_personas(path: str) -> Dict[str, Persona]:
    import yaml
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    out = {}
    for p in raw:
        out[p["id"]] = Persona(**p)
    return out

def save_run_md(conv, base_dir: str) -> str:
    runs = pathlib.Path(base_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    fname = runs / f"{conv.feature_id}_{conv.persona.id}.md"
    fname.write_text(conv.to_markdown())
    return str(fname)

def save_run_json(conv, base_dir: str) -> str:
    runs = pathlib.Path(base_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    fname = runs / f"{conv.feature_id}_{conv.persona.id}.json"
    from dataclasses import asdict
    data = asdict(conv)
    data["persona"] = asdict(conv.persona)
    data["messages"] = [asdict(m) for m in conv.messages]
    fname.write_text(json.dumps(data, indent=2))
    return str(fname)
