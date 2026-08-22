GENERATION_SYSTEM = """You are the grounded generation module of a retrieval system.
You answer ONLY from the numbered evidence blocks provided. Rules:
- Every factual claim must cite the evidence block(s) it comes from using their [n] indices.
- If the evidence does not contain the answer, say so explicitly in `answer` and explain what is missing in `uncertainty`.
- Never invent facts, numbers, names, or citations.
- Decompose your answer into atomic claims, each with the citation indices that support it and a confidence in [0,1].
- self_confidence: your overall confidence in [0,1] that the answer is correct given the evidence.

Evidence blocks are prefixed like: [1] filename | section path | page."""


def build_generation_user(
    query: str,
    context_text: str,
    history: list[tuple[str, str]],
    memory_notes: list[str],
) -> str:
    parts = []
    if memory_notes:
        parts.append("Known user/project context (may be relevant):\n" + "\n".join(f"- {note}" for note in memory_notes))
    if history:
        recent = "\n".join(f"{role.upper()}: {content[:300]}" for role, content in history[-4:])
        parts.append(f"Recent conversation:\n{recent}")
    parts.append(f"Evidence blocks:\n\n{context_text or '(no evidence retrieved)'}")
    parts.append(f"Question: {query}")
    return "\n\n".join(parts)
