
from emotion_app import emotion_predictor
import pytest

def test_format_and_keys():
    out = emotion_predictor("I am very happy today!")
    assert set(out.keys()) == {"anger","disgust","fear","joy","sadness","dominant_emotion","source"}
    assert isinstance(out["joy"], float)
    assert isinstance(out["dominant_emotion"], str)

def test_dominant_joy():
    out = emotion_predictor("I am happy and glad")
    assert out["dominant_emotion"] == "joy"

def test_dominant_sadness():
    out = emotion_predictor("I feel sad and depressed")
    assert out["dominant_emotion"] == "sadness"

def test_empty_raises():
    with pytest.raises(ValueError):
        emotion_predictor("   ")
