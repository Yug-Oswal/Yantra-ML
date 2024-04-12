from flask import Flask, request
import tensorflow as tf 
import keras_nlp
import tensorflow_hub as hub

preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    None,
    sequence_length=256,
)
encoder = keras_nlp.models.DistilBertBackbone.from_preset(
    None
)
encoder.trainable = True
txt = tf.keras.layers.Input(shape=(), dtype=tf.string)
x = preprocessor(txt)
x = encoder(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(Y_train.shape[1], activation='softmax')(x)
model = tf.keras.Model(inputs=[txt], outputs=x)
model.load_weights('../DistilBERT/EmotionExtractor')

app = Flask(__name__)

@app.post("/emotion-extract")
def extract_emotions():
    content = request.json['content']
    return "Works"
