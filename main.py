from flask import Flask, request, jsonify, Response
from mlx_lm import load, generate
from datetime import datetime
import time
import json

app = Flask(__name__)

MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit"
model, tokenizer = load(MODEL_NAME)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json

    print(f"Received request: {data}")

    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    stream = data.get("stream", True)

    start = time.time()
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=1000)
    print(f"Generated response in {time.time() - start: .2f}s")

    if not stream:
        return jsonify({
            "model": MODEL_NAME,
            "done": True,
            "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "message": {
                "role": "assistant",
                "content": response
            }
        })
    else:
        def res_generate():
            messages = [
                {"content": response.split('</think>\n\n')[-1], "done": True}
            ]

            for msg in messages:
                data = {
                    "model": MODEL_NAME,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "message": {
                        "role": "assistant",
                        "content": msg["content"],
                        "images": None
                    },
                    "done": msg["done"]
                }
                yield f"{json.dumps(data)}\n\n"
                time.sleep(0.1)  # Simulate processing time

        return Response(
            res_generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Transfer-Encoding': 'chunked'
            }
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
