#!/usr/bin/env python3
"""
Process conversations.json into cleaned data for the UI viewer.
Outputs: conversations_viewer/index.json + conversations_viewer/topics/*.json
Preserves the original file unchanged.
"""

import json
import re
import os
from collections import defaultdict

INPUT_FILE = "conversations.json"
OUTPUT_DIR = "conversations_viewer"
TOPICS_DIR = os.path.join(OUTPUT_DIR, "topics")

TOPIC_RULES = [
    ("AI & Machine Learning", [
        "claude", "gpt", "llm", "large language model", "neural network", "machine learning",
        "deep learning", "pytorch", "tensorflow", "langchain", "openai", "hugging face",
        "transformer", "fine-tun", "embedding", "rag ", "vector database", "diffusion model",
        "stable diffusion", "chatgpt", "gemini", "mistral", "anthropic", "artificial intelligence",
        " ai ", "ai model", "reinforcement learning", "natural language", "computer vision",
        "prompt engineering", "token limit", "inference", "training data", "ai agent",
        "multimodal", "claude code", "copilot", "perplexity", "midjourney", "dall-e",
        "autoregressive", "vae ", "variational autoencoder", "gan ", "generative",
        "silu activation", "leakyrelu", "batch normalization", "attention mechanism",
        "lora ", "qlora", "peft", "quantiz", "llama ", "mistral", "phi model",
        "huggingface", "groq", "tenstorrent", "sambanova", "cerebras",
        "physics-informed neural", "language model", "foundation model",
        "agentic", "cursor ", "windsurf", "replit", " nlp ", "sentiment analysis",
        "image generation", "text-to-image", "text generation",
    ]),
    ("Finance & Business", [
        "stock ", "equity ", "valuation", "startup", "venture capital", " vc ", "vc fund",
        "revenue", "profit", "earnings", "market cap", " ipo", "series a", "series b",
        "hedge fund", "investment bank", "private equity", "financial model",
        "balance sheet", "income statement", "cash flow", "ebitda", "p/e ratio",
        "nasdaq", "s&p 500", "dow jones", "portfolio", "dividend", "interest rate",
        "federal reserve", "inflation", "gdp", "macroeconomic", "fiscal", "monetary",
        "crypto", "bitcoin", "ethereum", "defi", "nft", "blockchain",
        "trading", "options ", "futures ", "derivatives",
        "business model", "monetiz", "saas ", " arr ", " mrr ", "churn",
        "profitability", "ipo ", "acquisition", "merger", "buyout",
        "company ", "corporation", "firm ", "enterprise", "ceo", "cfo",
        "market share", "competitive", "industry", "sector",
        "annual revenue", "quarterly", "fiscal year", "shareholder",
        "palantir", "nvidia", "apple ", "google", "microsoft", "amazon", "meta ",
        "netflix", "tesla", "uber", "airbnb", "stripe", "databricks", "snowflake",
        "openai valuation", "anthropic ", "groq ", "cohere ",
        "fund ", "investor", "founder", "accelerator", "incubator",
        "pitch deck", "term sheet", "cap table", "dilution",
        "intuit", "salesforce", "oracle", "sap", "workday", "servicenow",
        "applovin", "datadog", "crowdstrike", "cloudflare",
        "kalshi", "robinhood", "coinbase", "binance",
        "cost of ", "expense", "pricing ", "subscription",
        "market size", "tam ", "addressable market",
    ]),
    ("Programming & Software Dev", [
        "python", "javascript", "typescript", "java ", "c++ ", " c ", "golang", " rust ",
        "swift ", "kotlin", "react ", "vue ", "angular", "node.js", "django", "flask", "fastapi",
        "docker", "kubernetes", "git ", "github", "rest api", "graphql",
        "sql ", "nosql", "mongodb", "postgresql", "redis", "algorithm",
        "data structure", "debugging", "refactor", "unit test",
        "object-orient", "async ", "concurrent", "threading", "memory leak",
        "compile", "runtime error", "leetcode", "coding", "programming",
        "software engineer", "web scraping", "html ", "css ", "bash ", "shell script",
        "regex", "webpack", "vite", "npm ", "pip ", "conda ",
        "function ", "class ", "method ", "variable", "loop ", "recursion",
        "linked list", "binary tree", "hash map", "sorting",
        "matlab ", "fortran", "assembly", "cuda ", "opencl",
        "rewriting", "implementing", "fixing ", "error in code", "code review",
        "import ", "library", " api ", "sdk ", "framework",
    ]),
    ("Mathematics", [
        "calculus", "differential equation", "linear algebra", "matrix ",
        "eigenvalu", "eigendecomp", "probability", "statistics",
        "hypothesis test", "t-test", "p-value", "regression", "anova",
        "integral", "derivative", "gradient descent", "fourier", "laplace",
        "runge-kutta", "degrees of freedom", "standard deviation",
        "variance", "distribution", "normal distribution",
        "optimization problem", "convex optimiz", "math problem", "proof ",
        "theorem", "topology", "abstract algebra", "number theory",
        "combinatorics", "graph theory", "numerical method", "monte carlo",
        "bayesian", "markov", "stochastic", "set theory",
        "multivariable", "partial derivative", "jacobian", "hessian",
        "nash equilibrium", "game theory",
        "trigonometry", "geometry", "euclidean", "linear regression",
        "logistic regression", "pca ", "dimensionality reduction",
        "matrix multiplication", "dot product", "cross product",
        "hyperbolic geometry", "lorenz", "fixed point",
    ]),
    ("Science & Research", [
        "physics", "quantum", "relativity", "thermodynamic", "optics", "electromagnetism",
        "chemistry", "molecule", "chemical reaction", "compound", "element",
        "biology", "genetics", "dna", "protein", "cell biology", "evolution", "ecology",
        "neuroscience", "cognitive science", "climate change", "astronomy", "cosmology",
        "particle physics", "nuclear", "plasma", "fluid dynamics", "materials science",
        "research paper", "peer review", "experiment", "hypothesis",
        "geology", "paleontology", "oceanography", "meteorology",
        "photogrammetry", "spectroscopy", "electron cloud", "photon",
        "biodiversity", "ecosystem", "conservation",
        "netcdf", ".nc file", "climate data", "weather data",
        "lorenz 63", "lorenz system", "chaotic",
    ]),
    ("Technology & Engineering", [
        "robot", "hardware", "circuit", "electrical engineer", "mechanical engineer",
        "aerospace", "sensor", "microcontroller", "arduino", "raspberry pi",
        "iot", "embedded system", "fpga", "pcb", "3d print", "cad ",
        "drone", "autonomous vehicle", "self-driving", "lidar",
        "networking", "tcp/ip", "cybersecurity", "encryption", "firewall",
        "cloud computing", "aws ", "azure ", "gcp ", "serverless",
        "database design", "system design", "architecture", "infrastructure",
        "gpu ", "cpu ", "processor", "semiconductor", "chip design",
        "static analysis", "compiler design", "operating system",
        "mine removal", "demining", "military tech",
        "photovoltaic", "solar panel", "battery tech",
    ]),
    ("Health & Medicine", [
        "medical", "health ", "treatment", "medication", "surgery",
        "symptom", "diagnosis", "disease", "therapy", "clinical", "patient",
        "pharmaceutical", "vaccine", "immune system", "cancer", "cardiovascular",
        "mental health", "anxiety", "depression", "psychology", "psychiatry",
        "nutrition", "diet plan", "exercise", "fitness", "sleep ",
        "doctor", "hospital", "healthcare", "fda", "nih",
        "bowel", "digestion", "orthopedic", "food poisoning",
        "pharmacist", "prescription", "dosage",
        "therapist", "therapy platform", "headway",
    ]),
    ("Politics & Current Events", [
        "election", "political", "congress", "senate", "democrat", "republican",
        "government", "policy ", "regulation", "legislation", "supreme court",
        "president", "administration", "foreign policy", "geopolitic",
        "war ", "military", "nato", "ukraine", "middle east", "sanction",
        "immigration", "tariff", "trade war",
        "news ", "current event", "breaking news",
        "medicare for all", "repeal", "19th amendment",
        "federal budget", "national debt", "social security",
        "petition", "protest", "activist",
    ]),
    ("Education & Academia", [
        "homework", "assignment", "exam", "quiz", "grade ", "course ", "lecture",
        "university", "college", "school", "professor", "student",
        "textbook", "syllabus", "gpa", "study guide", "final exam",
        "dissertation", "thesis", "essay ", "cite", "citation",
        "sat ", "act ", "gre ", "mcat", "lsat",
        "am160", "hw1", "hw2", "exam part", "problem set",
        "degrees of freedom", "lecture notes",
        "eviction notice", "scholarship", "financial aid",
    ]),
    ("Entertainment & Media", [
        "movie", "film ", "tv show", "series ", "netflix", "hbo ", "disney",
        "music ", "song ", "album ", "artist ", "spotify", "youtube",
        "video game", "gaming", "esport", "twitch", "steam ",
        "book ", "novel", "author", "fiction",
        "sport", "football", "basketball", "soccer", "baseball", "tennis", "nba", "nfl",
        "celebrity", "actor ", "director ", "anime", "manga",
        "spider-man", "marvel", "shang-chi", "moana", "greatest showman",
        "sabrina carpenter", "3blue1brown", "edc ", "music festival",
        "box office", "streaming", "podcast",
        "mlb ", "japanese baseball", "nhl ", "mlb",
    ]),
    ("History & Culture", [
        "history of", "historical", "ancient", "civilization", "world war",
        "revolution", "empire", "dynasty", "medieval", "renaissance",
        "culture", "cultural", "tradition", "religion", "philosophy",
        "ethics", "moral ", "virtue", "existential",
        "language", "linguistics", "etymology", "fallacy",
        "mrs. lincoln", "mrs lincoln fallacy",
    ]),
    ("Personal & Career", [
        "food ", "recipe", "restaurant", "cooking", "diet",
        "travel", "vacation", "trip to",
        "career", "job ", "resume", "interview prep", "salary",
        "relationship", "dating", "social skill",
        "productivity", "habit", "time management",
        "home ", "apartment", "rent ", "mortgage",
        "fashion", "clothing", "shopping",
        "camping", "outdoor", "hiking",
        "rejection", "failure", "overcom",
        "stress", "burnout", "work-life",
        "recruiting", "hiring", "hr ",
        "recommendation letter", "reference letter",
    ]),
]

def classify_topic(name: str, all_human_text: str) -> str:
    text = (name + " " + all_human_text[:1000]).lower()
    scores = []
    for topic, keywords in TOPIC_RULES:
        count = sum(1 for kw in keywords if kw in text)
        scores.append((count, topic))
    scores.sort(reverse=True)
    if scores[0][0] == 0:
        return "Other"
    return scores[0][1]

def clean_message(msg: dict) -> dict:
    text = msg.get("text", "")
    text = re.sub(r"```\nViewing artifacts.*?```", "[code artifact]", text, flags=re.DOTALL)
    return {
        "sender": msg["sender"],
        "text": text,
        "created_at": msg["created_at"][:19].replace("T", " "),
        "attachments": [
            {"name": a.get("file_name", ""), "type": a.get("file_type", "")}
            for a in msg.get("attachments", [])
        ],
    }

def process():
    print("Loading conversations.json...", flush=True)
    with open(INPUT_FILE, "r") as f:
        raw = json.load(f)

    print(f"Processing {len(raw)} conversations...", flush=True)
    os.makedirs(TOPICS_DIR, exist_ok=True)

    index_entries = []
    topic_buckets = defaultdict(list)

    for i, conv in enumerate(raw):
        if i % 500 == 0:
            print(f"  {i}/{len(raw)}...", flush=True)

        msgs = conv.get("chat_messages", [])
        human_parts = []
        total_chars = 0
        first_human = ""
        for m in msgs:
            if m["sender"] == "human":
                if not first_human:
                    first_human = m["text"]
                if total_chars < 1000:
                    human_parts.append(m["text"])
                    total_chars += len(m["text"])
                else:
                    break
        human_text = " ".join(human_parts)

        topic = classify_topic(conv["name"], human_text)

        cleaned_conv = {
            "uuid": conv["uuid"],
            "name": conv["name"],
            "created_at": conv["created_at"][:10],
            "updated_at": conv["updated_at"][:10],
            "message_count": len(msgs),
            "topic": topic,
            "messages": [clean_message(m) for m in msgs],
        }

        # Lightweight index entry (no messages)
        preview = first_human[:120].replace("\n", " ").strip()
        index_entries.append({
            "uuid": conv["uuid"],
            "name": conv["name"] or first_human[:60] or "Untitled",
            "created_at": conv["created_at"][:10],
            "topic": topic,
            "message_count": len(msgs),
            "preview": preview,
        })

        topic_buckets[topic].append(cleaned_conv)

    # Sort index by date descending
    index_entries.sort(key=lambda c: c["created_at"], reverse=True)

    # Write index.json
    topic_counts = {t: len(convs) for t, convs in topic_buckets.items()}
    all_topics = sorted(topic_counts.keys(), key=lambda t: -topic_counts[t])

    index_data = {
        "total": len(index_entries),
        "topics": all_topics,
        "topic_counts": topic_counts,
        "conversations": index_entries,
    }
    index_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, ensure_ascii=False)
    print(f"Wrote {index_path} ({os.path.getsize(index_path)/1024:.0f} KB)")

    # Write per-topic JSON files
    for topic, convs in topic_buckets.items():
        convs.sort(key=lambda c: c["created_at"], reverse=True)
        safe_name = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
        path = os.path.join(TOPICS_DIR, f"{safe_name}.json")
        with open(path, "w") as f:
            json.dump(convs, f, ensure_ascii=False)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {topic}: {len(convs)} convs → {safe_name}.json ({size_kb:.0f} KB)")

    print("\nDone!")
    print(f"Topic breakdown:")
    for t in all_topics:
        print(f"  {t}: {topic_counts[t]}")

if __name__ == "__main__":
    process()
