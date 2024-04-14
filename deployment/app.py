from flask import Flask, request
# import tensorflow as tf 
import os
# from fastai.text.all import *
from transformers.utils import logging as hf_logging
from transformers import pipeline
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
# import keras_nlp

# from datasets import load_dataset

# import os, warnings, torch

# from blurr.text.data.all import *
# from blurr.text.modeling.all import *

# warnings.simplefilter("ignore")
hf_logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
# from collections import Counter

positive_emotions = ['admiration', 'amusement',  'approval', 'caring', 'curiosity', 'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'realization', 'relief', 'surprise']
negative_emotions = ['anger', 'annoyance', 'confusion', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness']

# preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
#     "distil_bert_base_en_uncased",
#     sequence_length=128,
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
# x = tf.keras.layers.Dense(28, activation='softmax')(x)
# model = tf.keras.Model(inputs=[txt], outputs=x)
# model.load_weights('./EmotionExtractor/DistilBERT_GoEmotions')

# inf_learn = load_learner(fname='../NER-Model/model.pkl')
# model = tf.saved_model.load('../EmotionExtractor')

app = Flask(__name__)

def filter_on_emotions(group_list, emotion):
    filtered = []
    emotion_mapping = {
        'admiration': 'gratitude',
        'amusement': 'joy',
        'approval': 'optimism',
        'caring': 'love',
        'curiosity': 'excitement',
        'desire': 'excitement',
        'excitement': 'joy',
        'gratitude': 'joy',
        'joy': 'optimism',
        'love': 'joy',
        'optimism': 'joy',
        'pride': 'admiration',
        'realization': 'optimism',
        'relief': 'gratitude',
        'surprise': 'amusement',
        'neutral': 'curiosity',
        'anger': 'relief',
        'annoyance': 'amusement',
        'confusion': 'curiosity',
        'disappointment': 'optimism',
        'disapproval': 'approval',
        'disgust': 'admiration',
        'embarrassment': 'amusement',
        'fear': 'curiosity',
        'grief': 'relief',
        'nervousness': 'curiosity',
        'remorse': 'optimism',
        'sadness': 'joy'
    }

    for group in group_list:
        if (group["emotion"] == emotion_mapping[emotion]):
            filtered.append(group)
    if (len(filtered) == 0):
        return group_list
    return filtered

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
    # output = model.predict([content])
    # return {
    #     "emotion": emotions[np.argmax(output)],
    #     "score": output[np.argmax(output)]
    # }
    classi = classifier(content)
    preds = classi[0]
    emos = []
    scores = []
    count = 0
    for emo_profile in preds:
        emos.append(emo_profile["label"])
        scores.append(emo_profile["score"] * 100)
        count += 1
        if (count == 3):
            break
    return {
        "emotions": emos,
        "scores": scores
    } 


# @app.post("/ner-extract")
# def extract_locations():
#     content = request.json['content']
#     results = inf_learn.blurr_predict_tokens(items=[content])
#     return results[0]

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
    emotions = person['emotions']
    scores = person['scores']
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

@app.post("/get-health-index")
def get_hindex():
    person = request.json['person']
    hscore = person['happiness_score']
    emotions = person['emotions']
    scores = person['scores']

    for i in range(len(emotions)):
        for j in range(len(emotions[i])):
            if (emotions[i][j] in positive_emotions):
                hscore += scores[i][j]
            elif (emotions[i][j] == "neutral"):
                hscore += 0
            else:
                hscore -= scores[i][j]
    return {
        "happiness_score": hscore
    }