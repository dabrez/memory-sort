#!/usr/bin/env python3
"""
Extract all software project ideas from conversations (topic files) and projects.json.
Uses Gemini API (via OpenAI-compatible endpoint) to comprehensively identify every
distinct software idea, including multiple ideas per conversation, with detailed design
write-ups.

Outputs:
  conversations_viewer/projects_all.json   — every idea (including repeats)
  conversations_viewer/projects.json       — grouped/deduplicated

Requires: pip install openai
"""

import json
import os
import re
import asyncio
import time
from collections import Counter
from openai import AsyncOpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

TOPICS_DIR = "conversations_viewer/topics"
PROJECTS_SOURCE_FILE = "projects.json"
OUTPUT_ALL = "conversations_viewer/projects_all.json"
OUTPUT_GROUPED = "conversations_viewer/projects.json"

MODEL = "gemini-3-flash-preview"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_CONCURRENT = 50   # conversations in flight at once
MAX_RPM = 100         # conservative — increase if you're on a higher tier
BATCH_SIZE = 25      # conversations per API call
CHECKPOINT_FILE = "conversations_viewer/checkpoint_ideas.json"


class RateLimiter:
    """Token-bucket rate limiter that caps requests per minute."""
    def __init__(self, rpm: int):
        self._rate = rpm / 60.0       # tokens per second
        self._max_tokens = float(rpm)
        self._tokens = float(rpm)     # start full
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + (now - self._last) * self._rate,
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)

# Topics to always fully process (highest idea density)
HIGH_YIELD_TOPICS = {
    "programming_software_dev",
    "technology_engineering",
    "ai_machine_learning",
    "finance_business",
    "science_research",
}

# Topics to still scan (may have product/tool ideas)
MEDIUM_YIELD_TOPICS = {
    "other",
    "personal_career",
    "education_academia",
    "health_medicine",
}

# Topics to skip unless conversation title looks project-related
LOW_YIELD_TOPICS = {
    "entertainment_media",
    "history_culture",
    "mathematics",
    "politics_current_events",
}

IDEA_NAME_KEYWORDS = [
    "startup", "idea", "project", "platform", "tool", "app",
    "system", "saas", "service", "bot", "api", "extension",
    "solver", "builder", "generator", "tracker", "manager",
    "innovative", "disrupt", "open source", "alternative",
    "building", "creating", "making", "developing", "product",
]

SKIP_WORDS = [
    "homework", " hw ", "essay", "exam", "midterm", "final exam",
    "lecture notes", "studying for", "reviewing for",
]

EXTRACTION_SYSTEM_PROMPT = """You are an expert at identifying software project ideas from conversations.

Your job: analyze the conversation and extract EVERY distinct software project idea mentioned.

Be thorough — include:
- Ideas that are the main topic of the conversation
- Ideas mentioned briefly or as examples
- Startup ideas, app ideas, tool ideas, API ideas, platform ideas
- Open source project ideas
- Ideas for solving specific problems with software
- Multiple separate ideas if the conversation covers several
- Ideas mentioned in passing, hypothetically, or as comparisons

For EACH idea, produce a JSON object:
{
  "name": "Short memorable project name (2-5 words, invent a name if none given)",
  "tagline": "One-line value proposition (under 90 chars)",
  "category": "One of: Developer Tools | Consumer Apps | Enterprise Software | AI/ML Tools | Healthcare Tech | FinTech | Education Tech | Social/Community | Infrastructure | Security | Media/Entertainment | Government/Civic | Other",
  "status": "idea",
  "description": "2-3 sentences: what the project does, who it's for, what problem it solves",
  "design_writeup": "300-500 word technical design covering: core functionality, recommended tech stack and why, key architecture decisions, main technical challenges, MVP scope vs future features, what makes this viable/unique",
  "technologies": ["relevant", "technologies", "and", "frameworks"],
  "source_conversation": "exact conversation title"
}

Return ONLY a valid JSON array. Empty array [] if no software project ideas found.
No markdown, no code fences, no text outside the JSON."""

BATCH_EXTRACTION_SYSTEM_PROMPT = """You are an expert at identifying software project ideas from conversations.

You will receive multiple conversations separated by "---". For EACH conversation extract every distinct software project idea mentioned (startup, app, tool, API, platform, open source project, etc.).

Return a JSON array with one entry per conversation:
[
  {
    "index": 0,
    "ideas": [
      {
        "name": "Short memorable project name (2-5 words)",
        "tagline": "One-line value proposition (under 90 chars)",
        "category": "Developer Tools | Consumer Apps | Enterprise Software | AI/ML Tools | Healthcare Tech | FinTech | Education Tech | Social/Community | Infrastructure | Security | Media/Entertainment | Government/Civic | Other",
        "status": "idea",
        "description": "2-3 sentences: what it does, who it's for, what problem it solves",
        "design_writeup": "200-400 word technical design: core functionality, tech stack, architecture, challenges, MVP scope",
        "technologies": ["relevant", "tech"],
        "source_conversation": "exact conversation title"
      }
    ]
  }
]

Use "ideas": [] for conversations with no software project ideas.
Return ONLY valid JSON, no markdown, no code fences."""

GROUPING_SYSTEM_PROMPT = """You are grouping a list of software project ideas that may describe the same or very similar projects.

For each cluster of similar ideas, identify:
1. A canonical group name (the clearest/best name for this idea)
2. Which idea IDs belong to the group (by index in the array)

Two ideas should be grouped if they describe the same core product/concept, even if named differently or with different scope details.

Return a JSON array of group objects:
[
  {
    "canonical_name": "Best name for this cluster",
    "indices": [0, 3, 7]
  }
]

Ideas that don't match any other idea should still appear as single-item groups.
Return ONLY the JSON array, no other text."""


def load_checkpoint() -> tuple[set[str], list[dict]]:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            data = json.load(open(CHECKPOINT_FILE))
            ids = set(data.get("processed_ids", []))
            ideas = data.get("ideas", [])
            print(f"Resuming from checkpoint: {len(ids)} convs done, {len(ideas)} ideas so far")
            return ids, ideas
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    return set(), []


def save_checkpoint(processed_ids: set[str], ideas: list[dict]):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_ids": list(processed_ids), "ideas": ideas}, f, ensure_ascii=False)


async def extract_ideas_from_conv(
    client: AsyncOpenAI,
    conv_name: str,
    conv_text: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> list[dict]:
    for attempt in range(5):
        try:
            async with semaphore:
                await rate_limiter.acquire()
                response = await client.chat.completions.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Title: {conv_name}\n\n{conv_text}"},
                    ],
                )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            ideas = json.loads(raw)
            if not isinstance(ideas, list):
                return []
            for idea in ideas:
                idea.setdefault("source_conversation", conv_name)
            return ideas
        except RateLimitError:
            wait = min(30 * (2 ** attempt), 300)
            print(f"  Rate limited on '{conv_name}', waiting {wait}s (attempt {attempt+1}/5)")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"  Error for '{conv_name}': {type(e).__name__}: {e}")
            return []
    print(f"  Gave up on '{conv_name}' after 5 rate limit retries")
    return []


def should_process_conv(conv: dict, topic_key: str) -> bool:
    name_lower = conv["name"].lower()

    # Always skip obvious homework/essay titles
    if any(kw in name_lower for kw in SKIP_WORDS):
        return False

    # High-yield topics: process everything
    if topic_key in HIGH_YIELD_TOPICS:
        return True

    # Medium-yield: process unless clearly non-project
    if topic_key in MEDIUM_YIELD_TOPICS:
        return True

    # Low-yield topics: only process if title suggests a project idea
    name_score = sum(1 for kw in IDEA_NAME_KEYWORDS if kw in name_lower)
    return name_score >= 1


def conv_to_text(conv: dict) -> str:
    msgs = conv.get("messages", [])
    parts = []
    for m in msgs[:60]:
        sender = "User" if m["sender"] == "human" else "Assistant"
        text = m.get("text", "")[:800]
        parts.append(f"{sender}: {text}")
    return "\n\n".join(parts)


def conv_to_text_compact(conv: dict) -> str:
    msgs = conv.get("messages", [])
    parts = []
    for m in msgs[:20]:
        sender = "User" if m["sender"] == "human" else "Assistant"
        text = m.get("text", "")[:400]
        parts.append(f"{sender}: {text}")
    return "\n\n".join(parts)


async def extract_ideas_from_batch(
    client: AsyncOpenAI,
    batch: list[tuple[str, str]],
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> list[list[dict]]:
    sections = [
        f"[Conversation {i}] Title: {name}\n\n{text}"
        for i, (name, text) in enumerate(batch)
    ]
    combined = "\n\n---\n\n".join(sections)

    for attempt in range(5):
        try:
            async with semaphore:
                await rate_limiter.acquire()
                response = await client.chat.completions.create(
                    model=MODEL,
                    max_tokens=16384,
                    messages=[
                        {"role": "system", "content": BATCH_EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": combined},
                    ],
                )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            results = json.loads(raw)
            if not isinstance(results, list):
                return [[] for _ in batch]
            ideas_by_index: dict[int, list[dict]] = {}
            for item in results:
                idx = item.get("index", 0)
                ideas = item.get("ideas", [])
                if isinstance(ideas, list) and 0 <= idx < len(batch):
                    for idea in ideas:
                        idea.setdefault("source_conversation", batch[idx][0])
                    ideas_by_index[idx] = ideas
            return [ideas_by_index.get(i, []) for i in range(len(batch))]
        except RateLimitError:
            wait = min(30 * (2 ** attempt), 300)
            print(f"  Rate limited on batch, waiting {wait}s (attempt {attempt+1}/5)")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"  Batch error: {type(e).__name__}: {e}")
            return [[] for _ in batch]
    print(f"  Gave up on batch after 5 rate limit retries")
    return [[] for _ in batch]


def make_id(name: str, idx: int = 0) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:55]
    return f"{slug}-{idx}" if idx else slug


def generate_claude_prompt(idea: dict) -> str:
    name = idea.get("name", "")
    desc = idea.get("description", "")
    design = idea.get("design_writeup", "")[:500]
    techs = idea.get("technologies", [])
    tech_str = ", ".join(techs[:8]) if techs else "TBD"
    return (
        f"Build {name}: {desc}\n\n"
        f"Design:\n{design}\n\n"
        f"Stack: {tech_str}\n\n"
        f"Start with the project structure and core MVP module. "
        f"Show folder layout, key files, and implement the main entry point."
    )


async def group_ideas_batch(
    client: AsyncOpenAI,
    ideas: list[dict],
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> list[dict]:
    """Use Gemini to cluster similar ideas. Process in batches of 60."""
    if not ideas:
        return []

    async def group_batch(batch: list[dict], offset: int):
        summary_list = "\n".join(
            f"{i}. {item.get('name','?')}: {item.get('tagline','')[:80]}"
            for i, item in enumerate(batch)
        )
        for attempt in range(5):
            try:
                async with semaphore:
                    await rate_limiter.acquire()
                    response = await client.chat.completions.create(
                        model=MODEL,
                        max_tokens=4096,
                        messages=[
                            {"role": "system", "content": GROUPING_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Group these {len(batch)} project ideas:\n\n{summary_list}"},
                        ],
                    )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r'^```(?:json)?\n?', '', raw)
                raw = re.sub(r'\n?```$', '', raw)
                groups = json.loads(raw)
                for g in groups:
                    g["indices"] = [i + offset for i in g["indices"]]
                return groups
            except RateLimitError:
                wait = min(30 * (2 ** attempt), 300)
                print(f"  Grouping rate limited, waiting {wait}s (attempt {attempt+1}/5)")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  Grouping error: {e}")
                break
        return [{"canonical_name": ideas[i + offset].get("name", ""), "indices": [i + offset]}
                for i in range(len(batch))]

    all_groups = []
    batch_size = 60
    for i in range(0, len(ideas), batch_size):
        batch = ideas[i:i+batch_size]
        groups = await group_batch(batch, i)
        all_groups.extend(groups)
        if i + batch_size < len(ideas):
            await asyncio.sleep(2)
    return all_groups


def build_grouped_projects(all_ideas: list[dict], groups: list[dict]) -> list[dict]:
    """Merge ideas within each group into a single canonical project entry."""
    result = []
    used = set()

    for group in groups:
        indices = [i for i in group.get("indices", []) if i < len(all_ideas)]
        if not indices:
            continue
        if all(i in used for i in indices):
            continue
        for i in indices:
            used.add(i)

        members = [all_ideas[i] for i in indices]
        canonical = max(members, key=lambda x: len(x.get("design_writeup", "")))

        all_sources = []
        for m in members:
            for sc in m.get("source_conversations", []):
                if sc and sc not in all_sources:
                    all_sources.append(sc)

        all_techs = []
        seen_techs = set()
        for m in members:
            for t in m.get("technologies", []):
                if t.lower() not in seen_techs:
                    all_techs.append(t)
                    seen_techs.add(t.lower())

        entry = {
            "id": make_id(group.get("canonical_name") or canonical.get("name", "unknown")),
            "name": group.get("canonical_name") or canonical.get("name", ""),
            "tagline": canonical.get("tagline", ""),
            "category": canonical.get("category", "Other"),
            "status": canonical.get("status", "idea"),
            "description": canonical.get("description", ""),
            "design_writeup": canonical.get("design_writeup", ""),
            "technologies": all_techs[:15],
            "source_conversations": all_sources,
            "mention_count": len(members),
            "claude_prompt": generate_claude_prompt(canonical),
        }
        result.append(entry)

    # Add any ungrouped ideas
    for i, idea in enumerate(all_ideas):
        if i not in used:
            entry = dict(idea)
            entry["id"] = make_id(idea.get("name", f"idea-{i}"))
            result.append(entry)

    return result


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    rate_limiter = RateLimiter(MAX_RPM)

    # ── Load conversations from topic files ────────────────────────────────────
    print("Loading conversations from topic files...")
    all_conversations = []
    topics_path = TOPICS_DIR
    for fname in sorted(os.listdir(topics_path)):
        if not fname.endswith(".json"):
            continue
        topic_key = fname[:-5]
        fpath = os.path.join(topics_path, fname)
        convs = json.load(open(fpath))
        print(f"  {fname}: {len(convs)} conversations")
        for conv in convs:
            conv["_topic_key"] = topic_key
        all_conversations.extend(convs)

    print(f"\nTotal conversations loaded: {len(all_conversations)}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    processed_ids, all_ideas = load_checkpoint()

    # ── Filter candidates ──────────────────────────────────────────────────────
    candidates = [c for c in all_conversations if should_process_conv(c, c.get("_topic_key", ""))]
    # Skip already-processed conversations
    pending = [c for c in candidates if c.get("uuid", c["name"]) not in processed_ids]
    print(f"Conversation candidates: {len(candidates)} total, {len(pending)} pending")
    topic_breakdown = Counter(c.get("topic", "Unknown") for c in pending)
    for topic, count in sorted(topic_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")

    # ── Extract from conversations ─────────────────────────────────────────────
    batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    print("\nExtracting ideas from conversations...")
    print(f"  {len(pending)} conversations → {len(batches)} batches of {BATCH_SIZE}, {MAX_CONCURRENT} concurrent, {MAX_RPM} RPM")
    est_min = len(batches) / MAX_RPM
    print(f"  Estimated time: ~{est_min:.0f} min (vs ~{len(pending)/MAX_RPM:.0f} min unbatched)")

    for i in range(0, len(batches), MAX_CONCURRENT):
        batch_group = batches[i:i + MAX_CONCURRENT]
        tasks = [
            extract_ideas_from_batch(
                client,
                [(c["name"], conv_to_text_compact(c)) for c in batch],
                semaphore,
                rate_limiter,
            )
            for batch in batch_group
        ]
        batch_results = await asyncio.gather(*tasks)
        for batch, per_conv_ideas in zip(batch_group, batch_results):
            for conv, ideas in zip(batch, per_conv_ideas):
                all_ideas.extend(ideas)
                processed_ids.add(conv.get("uuid", conv["name"]))
        done_convs = min((i + len(batch_group)) * BATCH_SIZE, len(pending))
        print(f"  {done_convs}/{len(pending)} convs processed, {len(all_ideas)} ideas so far")
        save_checkpoint(processed_ids, all_ideas)

    # ── Extract from Claude projects.json ──────────────────────────────────────
    print(f"\nLoading Claude projects from {PROJECTS_SOURCE_FILE}...")
    try:
        projects_source = json.load(open(PROJECTS_SOURCE_FILE))
        print(f"  {len(projects_source)} Claude projects")
    except Exception as e:
        print(f"  Could not load {PROJECTS_SOURCE_FILE}: {e}")
        projects_source = []

    skip_proj_words = ["homework", " hw", "essay", "exam", "midterm", "lecture",
                       "philosophy", "reviewing", "studying", "math 1", "am 1",
                       "cse 101", "program0", "final essay", "how to use claude"]
    proj_tasks = []
    pending_projects = []
    for p in projects_source:
        name_lower = p.get("name", "").lower()
        if any(kw in name_lower for kw in skip_proj_words):
            continue
        if f"[Project] {p['name']}" in processed_ids:
            continue
        desc = p.get("description", "")
        docs = p.get("docs", [])
        doc_content = " ".join(d.get("content", "") for d in docs)
        text = f"Claude Project: {p['name']}\nDescription: {desc}\n{doc_content}"
        proj_tasks.append(
            extract_ideas_from_conv(client, f"[Project] {p['name']}", text, semaphore, rate_limiter)
        )
        pending_projects.append(p)

    print(f"Processing {len(proj_tasks)} relevant Claude projects...")
    proj_count = 0
    for i in range(0, len(proj_tasks), MAX_CONCURRENT):
        group_tasks = proj_tasks[i:i + MAX_CONCURRENT]
        group_projects = pending_projects[i:i + MAX_CONCURRENT]
        results = await asyncio.gather(*group_tasks)
        for p, ideas in zip(group_projects, results):
            all_ideas.extend(ideas)
            proj_count += len(ideas)
            processed_ids.add(f"[Project] {p['name']}")
        save_checkpoint(processed_ids, all_ideas)
    print(f"  Got {proj_count} ideas from Claude projects")

    print(f"\nTotal ideas extracted: {len(all_ideas)}")

    # ── Assign raw IDs and normalize ───────────────────────────────────────────
    for i, idea in enumerate(all_ideas):
        # Normalize source_conversations field
        sc = idea.pop("source_conversation", "") or ""
        existing = idea.get("source_conversations", [])
        if isinstance(existing, list):
            sources = existing
        else:
            sources = []
        if sc and sc not in sources:
            sources.append(sc)
        idea["source_conversations"] = sources
        idea["mention_count"] = 1
        idea["claude_prompt"] = generate_claude_prompt(idea)
        for field in ["name", "tagline", "category", "status", "description", "design_writeup"]:
            idea.setdefault(field, "")
        idea.setdefault("technologies", [])
        idea.setdefault("status", "idea")
        idea["id"] = make_id(idea.get("name", f"idea-{i}"), 0)

    # Make IDs unique
    id_counts: dict[str, int] = {}
    for idea in all_ideas:
        base = idea["id"]
        id_counts[base] = id_counts.get(base, 0) + 1
        if id_counts[base] > 1:
            idea["id"] = f"{base}-{id_counts[base]}"

    # ── Write all ideas (undeduped) ────────────────────────────────────────────
    all_ideas_sorted = sorted(all_ideas, key=lambda x: (x.get("category", ""), x.get("name", "")))
    with open(OUTPUT_ALL, "w") as f:
        json.dump(all_ideas_sorted, f, ensure_ascii=False, indent=2)
    size_kb = os.path.getsize(OUTPUT_ALL) / 1024
    print(f"\nWrote {OUTPUT_ALL} — {len(all_ideas_sorted)} ideas ({size_kb:.0f} KB)")

    # ── Group similar ideas ────────────────────────────────────────────────────
    print(f"\nGrouping {len(all_ideas)} ideas by similarity...")
    groups = await group_ideas_batch(client, all_ideas, semaphore, rate_limiter)
    grouped = build_grouped_projects(all_ideas, groups)

    grouped.sort(key=lambda x: (-x.get("mention_count", 1), x.get("name", "")))

    with open(OUTPUT_GROUPED, "w") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)
    size_kb = os.path.getsize(OUTPUT_GROUPED) / 1024
    print(f"Wrote {OUTPUT_GROUPED} — {len(grouped)} grouped projects ({size_kb:.0f} KB)")

    # Clean up checkpoint now that outputs are written successfully
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint cleared.")

    print("\nCategory breakdown (grouped):")
    cats = Counter(p.get("category", "Other") for p in grouped)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    print("\nTop projects by mention count:")
    for p in grouped[:20]:
        print(f"  [{p.get('mention_count',1)}x] {p.get('name','?')} — {p.get('tagline','')[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
