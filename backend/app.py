from flask import Flask, request, jsonify
from generate_image import generate_image
import os

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    output_file = data.get("output", "output.png")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    output_path = os.path.join("outputs", output_file)
    output_path = generate_image(prompt, output_path)

    return jsonify({"message": "Image generated", "output": output_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
