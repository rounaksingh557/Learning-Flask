from flask import Flask, render_template, request, jsonify, Response
from Model.Text_Sentiment import *

app = Flask(__name__)


@app.route('/')
def index() -> str:
    'This is the initial function of the app.'
    return render_template('index.html')


@app.route('/predict-emotion', methods=["POST"])
def predict_emotion() -> Response:
    input_text = request.json.get("text")
    if not input_text:
        response: dict = {
            "Status": "ERROR",
            "Message": "Please enter something to predict"
        }
        return jsonify(response)
    else:
        predicted_emotion, predicted_emotion_img_url = predict(input_text)
        response: dict = {
            "Status": "SUCCESS",
            "data": {
                "predicted_emotion": predicted_emotion,
                "predicted_emotion_img_url": predicted_emotion_img_url
            }
        }
        return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
