from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import cv2
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import  decode_predictions
from tensorflow.keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'E:/Fiverr/Machine Learning/19 Alzheimer Disease Classification/Alzheimer MRI Classification/saved model/my_model.h5' # set path to saved model here

# Load your trained model
model = load_model(MODEL_PATH)


print('Model loaded. Check http://127.0.0.1:5000/')

classes = {
    0:'Mild Demented',
    1:'Moderate Demented',
    2:'Non Demented',
    3:'Very Mild Demented'
}

def model_predict(img_path, model):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (208, 176))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    y_pred = np.argmax(preds, axis=1)
    #preds = model.predict_classes([image])[0]
    print(y_pred)
    return y_pred #preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction

        preds = model_predict(file_path, model)
        #sign = classes[preds]
        #result = str(sign)
        if preds == 0:
            labels = 'Mild Demented'
        elif preds== 1:
            labels = 'Moderate Demented'
        elif preds==2:
            labels = 'Non Demented'
        elif preds==3:
            labels = 'Very Mild Demented'

        return labels
    return None


if __name__ == '__main__':
    app.run(debug=True)

