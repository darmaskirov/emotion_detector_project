
"""Flask server for the Emotion Detector assignment."""
from __future__ import annotations
from typing import Tuple
from flask import Flask, jsonify, request
from emotion_app import emotion_predictor

def create_app() -> Flask:
    """Application factory: returns the configured Flask app."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        """Health/info endpoint."""
        return jsonify({"message": "Emotion Detector API is running."})

    @app.get("/emotionDetector")
    def emotion_detector_route():
        """Analyze the `textToAnalyze` query param and return emotion JSON."""
        text = request.args.get("textToAnalyze", type=str)
        if text is None or not str(text).strip():
            return jsonify({"error": "Text is required."}), 400
        try:
            result = emotion_predictor(text)
            return jsonify(result)
        except Exception as ex:  # pragma: no cover
            # Unexpected error → 500 with details hidden (safe message)
            return jsonify({"error": "Internal Server Error"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    # Run dev server
    app.run(debug=False)
