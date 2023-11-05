from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import openai
import requests
import io
from pydub import AudioSegment
from pydub.playback import play
from decouple import config
import speech_recognition as sr

app = Flask(__name__, static_folder='static')

openai.api_key = config('OPENAI_API_KEY')
ELEVEN_LABS_API_KEY = config('ELEVEN_LABS_API_KEY')
ELEVEN_LABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/piTKgcLEGmPE4e6mEKli/stream"

def text_to_speech(text):
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_LABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(ELEVEN_LABS_API_URL, json=data, headers=headers, stream=True)

    if response.status_code == 200:
        return response.content
    else:
        return None

def chat_with_gpt(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant, answer questions under 20 words."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message["content"]

recognizer = sr.Recognizer()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    gpt_response = chat_with_gpt(user_input)

    # Generate audio response and save it to a file
    gpt_audio = text_to_speech(gpt_response)
    with open(os.path.join(app.static_folder, "response.mp3"), "wb") as audio_file:
        audio_file.write(gpt_audio)

    response_data = {
        "responseText": gpt_response,
        "responseAudio": url_for("serve_audio", filename="response.mp3"),
    }

    return jsonify(response_data)

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)
