
"""Emotion detection module with optional IBM Watson NLU usage.
Falls back to a local keyword heuristic if Watson is unavailable.
"""
from __future__ import annotations
from typing import Dict
import os

# Optional Watson import
try:
    # pylint: disable=import-error
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
    _WATSON_AVAILABLE = True
except Exception:  # noqa: BLE001 - optional dependency
    _WATSON_AVAILABLE = False


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to 0..1 range (sum to 1 if any > 0)."""
    total = sum(scores.values())
    if total <= 0.0:
        return {k: 0.0 for k in scores}
    return {k: round(v / total, 2) for k, v in scores.items()}


def _dominant(scores: Dict[str, float]) -> str:
    """Return the max-scoring emotion label."""
    return max(scores.items(), key=lambda kv: kv[1])[0] if scores else "joy"


def _heuristic_predict(text: str) -> Dict[str, float]:
    """Very simple keyword-based heuristic for offline usage."""
    text_l = text.lower()
    lexicon = {
        "joy": ["happy", "joy", "glad", "delight", "love", "awesome", "great"],
        "sadness": ["sad", "unhappy", "depress", "cry", "down"],
        "anger": ["angry", "mad", "furious", "rage", "annoy"],
        "fear": ["fear", "scared", "afraid", "terrified", "anxious"],
        "disgust": ["disgust", "gross", "nasty", "repuls", "sickening"],
    }
    raw = {e: 0.0 for e in ["anger", "disgust", "fear", "joy", "sadness"]}
    for emotion, words in lexicon.items():
        for w in words:
            if w in text_l:
                raw[emotion] += 1.0
    return raw


def _watson_predict(text: str) -> Dict[str, float]:
    """Call IBM Watson NLU if credentials are available, else raise RuntimeError."""
    if not _WATSON_AVAILABLE:
        raise RuntimeError("Watson SDK is not installed.")
    api_key = os.getenv("WATSON_API_KEY")
    url = os.getenv("WATSON_URL")
    if not api_key or not url:
        raise RuntimeError("Watson credentials are not set in env vars.")
    authenticator = IAMAuthenticator(api_key)
    nlu = NaturalLanguageUnderstandingV1(version="2021-08-01", authenticator=authenticator)
    nlu.set_service_url(url)
    response = nlu.analyze(text=text, features=Features(emotion=EmotionOptions())).get_result()
    # Watson returns a structure like {'emotion': {'document': {'emotion': {...}}}}
    doc = response.get("emotion", {}).get("document", {}).get("emotion", {})
    # Map to our expected keys (anger, disgust, fear, joy, sadness)
    scores = {k: float(doc.get(k, 0.0)) for k in ["anger", "disgust", "fear", "joy", "sadness"]}
    return scores


def emotion_predictor(text: str) -> Dict[str, object]:
    """Return emotion scores and dominant emotion.

    Args:
        text: Input string to analyze.

    Returns:
        Dict with keys: anger, disgust, fear, joy, sadness, dominant_emotion, source

    Raises:
        ValueError: If text is empty/blank.
    """
    if text is None or not str(text).strip():
        raise ValueError("Text is required.")

    try:
        scores = _watson_predict(text)
        source = "watson"
    except Exception:
        scores = _heuristic_predict(text)
        source = "heuristic"

    scores = _normalize(scores)
    result: Dict[str, object] = {
        "anger": scores["anger"],
        "disgust": scores["disgust"],
        "fear": scores["fear"],
        "joy": scores["joy"],
        "sadness": scores["sadness"],
        "dominant_emotion": _dominant(scores),
        "source": source,
    }
    return result


# Backwards-compat alias used in some course texts
emotion_detector = emotion_predictor
