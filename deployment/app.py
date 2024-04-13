from flask import Flask, request
import tensorflow as tf 
import os
# from blurr.text.data.all import *
# from blurr.text.modeling.all import *
from collections import Counter

positive_emotions = ['admiration', 'amusement',  'approval', 'caring', 'curiosity', 'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'realization', 'relief', 'surprise', 'neutral']
negative_emotions = ['anger', 'annoyance', 'confusion', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness']

# preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
#     "distil_bert_base_en_uncased",
#     sequence_length=256,
# )
# encoder = keras_nlp.models.DistilBertBackbone.from_preset(
#     "distil_bert_base_en_uncased"
# )
# encoder.trainable = True
# txt = tf.keras.layers.Input(shape=(), dtype=tf.string)
# x = preprocessor(txt)
# x = encoder(x)
# x = tf.keras.layers.GlobalAveragePooling1D()(x)
# x = tf.keras.layers.Dropout(0.1)(x)
# x = tf.keras.layers.Dense(Y_train.shape[1], activation='softmax')(x)
# model = tf.keras.Model(inputs=[txt], outputs=x)
# model.load_weights('../DistilBERT/EmotionExtractor')

# model = tf.saved_model.load('../EmotionExtractor')
# # tf.config.set_visible_devices([], 'GPU')  
# predictor = model.signatures["serving_default"]

# inf_learn = load_learner(fname='./NER-Model/model.pkl')

app = Flask(__name__)

def filter_on_emotions(group_list, emotion):
    def find_groups(group_type):
        filtered = []
        if (group_type == "positive"):
            for group in group_list:
                if (group['emotion'] in positive_emotions):
                    filtered.append(group)
        elif (group_type == "negative"):
            for group in group_list:
                if (group['emotion'] in positive_emotions):
                    filtered.append(group)
        return filtered

    filtered_groups = []
    if (emotion in negative_emotions):
        filtered_groups = find_groups("positive")
    else: 
        filtered_groups = find_groups("positive")

def filter_on_location(group_list, location):
    filtered = []
    for group in group_list:
        if (group["location"] == location):
            filtered.append(group)
    if (len(filtered) == 0):
        return group_list
    return filtered

@app.post("/emotion-extract")
def extract_emotions():
    content = request.json['content']
    output = model.predict([content])
    return {
        "emotion": emotions[np.argmax(output)],
        "score": output[np.argmax(output)]
    }

@app.post("/ner-extract")
def extract_locations():
    content = request.json['content']
    results = inf_learn.blurr_predict_tokens(items=[content])
    return results[0]

@app.post("/get-dominating-emotion")
def get_dominating_emotion():
    ppl = [
        {
            "id": 1234,
            "emotions": ["joy", "love", "disgust", "sadness", "joy", "love", "love"]
        }
    ]
    all_emotions = []
    for person in ppl: 
        counts = Counter(person["emotions"])
        max_ele, max_count = counts.most_common(1)[0]
        all_emotions.append(max_ele)
    max_emo, max_count = Counter(all_emotions).most_common(1)[0]
    return {"dominatingEmotion": max_emo}


@app.post("/list-recommends")
def list_recommends():
    person = {
        "id": 1234,
        "emotions": [["joy", "sadness", "remorse"], ["excitement", "joy", "grief"]],
        "scores": [[7.2, 3.1, 1.2], [6.6, 5.5, 0.3]], 
        "locations": ["Bangaladesh", "Vietnam"]
    }
    d = {}
    for i in range(len(person['emotions'])):
        for j in range(len(emotions[i])): 
            if emotions[i][j] not in list(d.keys()):
                d[emotions[i][j]] = scores[i][j]
            else:
                d[emotions[i][j]] += scores[i][j]
    emotion = max(d, key=d.get)
    groups = [
        {
            "id": 123, 
            "emotion": "joy",
            "location": "Vietnam"
        }
    ]
    groups = filter_on_emotions(groups, emotion)
    for location in person['locations']:
        groups = filter_on_location(groups, location)
    return groups 