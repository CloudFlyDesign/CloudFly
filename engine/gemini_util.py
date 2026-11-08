import os
import time

from google import genai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
_client = None


def gemini_call(prompt, tries=5):
    """Call Gemini, retrying transient overload errors (503/429) with growing waits."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for i in range(tries):
        try:
            return _client.models.generate_content(model=MODEL, contents=prompt).text
        except Exception as e:
            if any(s in str(e) for s in ("503", "UNAVAILABLE", "429")):
                time.sleep(2 ** i)
                continue
            raise
    raise RuntimeError("Gemini unavailable after retries")