from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
#from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

#model._make_predict_function()      
print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    #update by ViPS
    img = cv2.imread(img_path)
    new_arr = cv2.resize(img,(100,100))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 100, 100, 3)
    

    
    preds = model.predict(new_arr)
    return preds


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
            basepath, 'uploads',f.filename )  #secure_filename(f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax()              # Simple argmax
 
        
        CATEGORIES = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
            'Potato___Early_blight' ,'Potato___Late_blight', 'Potato___healthy',
            'Tomato_Bacterial_spot' ,'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold' ,'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite' ,'Tomato__Target_Spot',
            'Tomato__YellowLeaf__Curl_Virus', 'Tomato_mosaic_virus',
            'Tomato_healthy']
        return CATEGORIES[pred_class]

        #return CATEGORIES[pred_class]
    return None


if __name__ == '__main__':
    app.run(debug=True)

