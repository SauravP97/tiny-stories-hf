# app.py
from flask import Flask, render_template, request, jsonify, stream_with_context, Response
from inference_streamer import stream_inference
import logging

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    model_size = data.get("model_size")
    logging.info('Model size selected: ', model_size)

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        def generate():
            # Iterate over the generator from your inference script
            for token in stream_inference(user_input):
                # We wrap the token in a JSON structure or send raw text
                # Sending raw text is often easier for simple streams
                yield token.encode('utf-8')
        return Response(stream_with_context(generate()), mimetype='text/plain')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)