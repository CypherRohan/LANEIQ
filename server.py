from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import requests

app = Flask(__name__)

GEMINI_API_KEY = "AIzaSyDp5K5XhEkw27O5X6UWxMSlaIpEZyHwWsA"

@app.route('/chat-gemini', methods=['POST'])
def chat_gemini():
    data = request.get_json()
    user_message = data.get('message')

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [{"text": user_message}]
        }]
    }

    response = requests.post(
        f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}',
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        bot_reply = response.json()['candidates'][0]['content']['parts'][0]['text']
        return jsonify({'message': bot_reply})
    else:
        return jsonify({'message': f'Error: {response.text}'}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load-model')
def load_model():
    return 'Model loaded successfully'

@app.route('/process-frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    
    image_data = np.array(data['image'], dtype=np.uint8)

    width = 1280
    height = 720
    image = image_data.reshape((height, width, 4))  # RGBA channels
    
    detections = [
        {"x": 100, "y": 150, "width": 80, "height": 80, "label": "Lane Drift", "confidence": 0.87},
        {"x": 300, "y": 400, "width": 120, "height": 100, "label": "Overspeed", "confidence": 0.92}
    ]
    
    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=True)
