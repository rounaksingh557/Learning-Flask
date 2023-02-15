import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

train_data = pd.read_csv("../static/data_files/tweet_emotions.csv")
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "content"]
    training_sentences.append(sentence)

model = load_model("../static/model_files/Tweet_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

# assign emoticons for different emotions
emo_code_url = {
    "empty": [0, "../static/emoticons/Empty.png"],
    "sadness": [1, "../static/emoticons/Sadness.png"],
    "enthusiasm": [2, "../static/emoticons/Enthusiasm.png"],
    "neutral": [3, "../static/emoticons/Neutral.png"],
    "worry": [4, "../static/emoticons/Worry.png"],
    "surprise": [5, "../static/emoticons/Surprise.png"],
    "love": [6, "../static/emoticons/Love.png"],
    "fun": [7, "../static/emoticons/fun.png"],
    "hate": [8, "../static/emoticons/hate.png"],
    "happiness": [9, "../static/emoticons/happiness.png"],
    "boredom": [10, "../static/emoticons/boredom.png"],
    "relief": [11, "../static/emoticons/relief.png"],
    "anger": [12, "../static/emoticons/anger.png"]

}


def predict(text: str):
    predicted_emotion: str = ""
    predicted_emotion_img_url: str = ""

    if text != "":
        sentence = []
        sentence.append(text)
        sequences = tokenizer.texts_to_sequences(sentence)
        padded = pad_sequences(sequences, max_length, padding_type, trunc_type)
        testing_padded = np.array(padded)
        predicted_class_label = np.argmax(
            model.predict(testing_padded), axis=1)
        print(predicted_class_label)
        for key, value in emo_code_url.items():
            if value[0] == predicted_class_label:
                predicted_emotion_img_url = value[1]
                predicted_emotion = key
        return predicted_emotion, predicted_emotion_img_url
    else:
        return None
