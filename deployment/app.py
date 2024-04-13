from flask import Flask, request
import tensorflow as tf 
import os
from blurr.text.data.all import *
from blurr.text.modeling.all import *

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

model = tf.saved_model.load('../EmotionExtractor')
# # tf.config.set_visible_devices([], 'GPU')  
# predictor = model.signatures["serving_default"]

inf_learn = load_learner(fname='./NER-Model/model.pkl')

app = Flask(__name__)

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
    return results