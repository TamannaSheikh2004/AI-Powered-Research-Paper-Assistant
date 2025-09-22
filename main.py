# main.py
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
if not MISTRAL_API_KEY:
    print("MISTRAL_API_KEY missing in .env")
    exit(1)

def generate_text_mistral(prompt: str, model: str = "mistral-large-latest", max_tokens: int = 1200):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    r = requests.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    j = r.json()
    try:
        return j["choices"][0]["message"]["content"].strip()
    except Exception:
        return json.dumps(j, indent=2)[:2000]

if __name__ == "__main__":
    topic = input("Enter your research topic: ").strip()
    if not topic:
        print("No topic entered.")
        exit(1)
    prompt = f"Generate a compact research-paper style draft on: {topic}\nUse headings: Abstract, Introduction, Literature Review, Methodology, Results, Discussion, Conclusion, References."
    print("Generating...")
    out = generate_text_mistral(prompt)
    print("\n===== DRAFT =====\n")
    print(out)
    with open("mistral_draft.txt", "w", encoding="utf-8") as wf:
        wf.write(out)
    print("\nSaved draft to mistral_draft.txt")