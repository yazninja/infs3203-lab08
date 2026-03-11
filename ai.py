import os, json, re
import requests

MODEL = "gemini-2.5-flash-lite"
API_ENDPOINT = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{MODEL}:generateContent"


def _call(prompt: str) -> str:
    api_key = os.environ["GEMINI_KEY"]
    url = f"{API_ENDPOINT}?key={api_key}"

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    response = requests.post(
        url, json=payload, headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()

    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


def _clean_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_questions(topic: str, difficulty: str, count: int = 5) -> list:
    prompt = f"""Generate {count} multiple-choice quiz questions about "{topic}" at {difficulty} difficulty.
Return ONLY a JSON array — no markdown, no extra text. Each element must have:
  "id": integer (1-based)
  "topic": "{topic}"
  "question": string
  "options": array of exactly 4 strings
  "answerIndex": integer 0-3 (index of the correct option)
  "explanation": one-sentence explanation of the correct answer"""
    return json.loads(_clean_json(_call(prompt)))


def explain_answer(
    question: str, options: list, answer_index: int, selected_index: int
) -> str:
    prompt = f"""A student answered a quiz question incorrectly.
Question: {question}
Options: {json.dumps(options)}
Correct answer: option {answer_index} — "{options[answer_index]}"
Student chose: option {selected_index} — "{options[selected_index]}"
Explain in 2-3 sentences why the correct answer is right and why the student's choice is wrong."""
    return _call(prompt).strip()


def recommend_study(topic_scores: list) -> str:
    prompt = f"""A student just completed a quiz. Results by topic:
{json.dumps(topic_scores, indent=2)}
Each entry has: topic, correct (number correct), total (number attempted).
Write a short personalised study recommendation (3-4 sentences) focusing on their weakest areas."""
    return _call(prompt).strip()
