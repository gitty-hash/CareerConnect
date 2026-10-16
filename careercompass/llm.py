"""One gateway for every LLM call (Chapter 3 tutorial pattern: a single chat() helper).

The course notebooks put all OpenAI traffic behind one function so that mock mode and
options live in exactly one place. CareerCompass keeps that design and replaces the
hand-written mock answers with a *response cache*:

- Live mode   (OPENAI_API_KEY set): call gpt-4o-mini and write the answer to the cache.
- Cached mode (no key, or forced):  replay the exact answer recorded for the same request.

The cache key is a hash of (model, messages, temperature, json_mode), so a cached answer
is only ever returned for the identical prompt that produced it. Requests that were never
recorded raise NotCached, and the UI says so honestly instead of inventing an answer.
"""

import hashlib
import json
import os
import threading
import time

from .config import LLM_CACHE_FILE, LLM_MODEL


class NotCached(Exception):
    """Raised in cached demo mode when a request was never recorded."""


_lock = threading.Lock()
_cache = None
_client = None
_force_cached = False


def _load_cache():
    global _cache
    if _cache is None:
        if LLM_CACHE_FILE.exists():
            _cache = json.loads(LLM_CACHE_FILE.read_text(encoding="utf-8"))
        else:
            _cache = {}
    return _cache


def _save_cache():
    LLM_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LLM_CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_cache, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(LLM_CACHE_FILE)


def api_key_available():
    # Best practice from the course setup cell: read the key from the environment,
    # never hard-code it.
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def set_force_cached(flag):
    """Let the UI switch to cached demo mode even when a key is present."""
    global _force_cached
    _force_cached = bool(flag)


def is_live():
    return api_key_available() and not _force_cached


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.environ["OPENAI_API_KEY"].strip()
        try:
            import httpx2
            # Same patch as the course notebooks: request uncompressed responses to work
            # around a Brotli/httpx2 decoder incompatibility ('APIConnectionError').
            _client = OpenAI(api_key=api_key,
                             http_client=httpx2.Client(headers={"Accept-Encoding": "identity"}))
        except ImportError:
            _client = OpenAI(api_key=api_key)
    return _client


def cache_key(messages, temperature, json_mode, model=LLM_MODEL):
    payload = json.dumps({"model": model, "messages": messages,
                          "temperature": temperature, "json_mode": json_mode},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_cached(messages, temperature=0.0, json_mode=False):
    return cache_key(messages, temperature, json_mode) in _load_cache()


def chat(messages, tag=None, temperature=0.0, json_mode=False, refresh=False):
    """One gateway for all LLM calls (Ch3). Cache first, then live API if allowed."""
    cache = _load_cache()
    key = cache_key(messages, temperature, json_mode)
    if key in cache and not (refresh and is_live()):
        return cache[key]["response"]
    if not is_live():
        raise NotCached(tag or "llm call")

    kwargs = {"model": LLM_MODEL, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _get_client().chat.completions.create(**kwargs).choices[0].message.content

    with _lock:
        cache[key] = {"tag": tag, "response": response, "model": LLM_MODEL,
                      "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        _save_cache()
    return response


def chat_json(messages, tag=None, temperature=0.0, refresh=False):
    """chat() with json_mode on, parsed into a dict (Ch3 Step 2: code reads the answer)."""
    return json.loads(chat(messages, tag=tag, temperature=temperature,
                           json_mode=True, refresh=refresh))


def cache_stats():
    cache = _load_cache()
    tags = {}
    for entry in cache.values():
        t = (entry.get("tag") or "other").split(":")[0]
        tags[t] = tags.get(t, 0) + 1
    return len(cache), tags
