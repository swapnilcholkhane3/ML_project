from flask import Flask, render_template, jsonify,redirect, url_for, request
import json 
import pyttsx3
import numpy as np
from keras.layers import Dense, LSTM, TimeDistributed, Embedding, Activation, RepeatVector, Concatenate
from keras.models import Sequential, Model
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing.sequence import pad_sequences
import os
import re
import cv2
import gtts
import csv


app = Flask(__name__)

with open('data.json', 'r') as jf:
    data = json.load(jf)


def load_model():
    vocabs = np.load('vocab.npy', allow_pickle=True)
    vocabs = vocabs.item()
    inv_vocabs = {v:k for k, v in vocabs.items()}
    embedding_size = 128
    vocab_size = len(vocabs)
    max_length = 40

    model = Sequential()
    model.add(Dense(embedding_size, input_shape=(2048, ), activation='relu'))
    model.add(RepeatVector(max_length))

    lang_model = Sequential()
    lang_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length))
    lang_model.add(LSTM(256, return_sequences=True))
    lang_model.add(TimeDistributed(Dense(embedding_size)))

    concat = Concatenate()([model.output, lang_model.output])
    lstm = LSTM(128, return_sequences=True)(concat)
    lstm = LSTM(512, return_sequences=False)(lstm)
    lstm = Dense(vocab_size)(lstm)
    output = Activation('softmax')(lstm)
    model_ = Model(inputs=[model.input, lang_model.input], outputs=output)
    model_.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    model_.load_weights('weights.h5')

    print('Model Loaded..')

    resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    return max_length, vocabs, inv_vocabs, model_, resnet_model


def predict_caption(max_length, vocabs, inv_vocabs, model, image, resnet_model, names, caps):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.reshape(image, (1, 224, 224, 3))
    incept = resnet_model.predict(image).reshape(1, 2048)

    text = ['startofseq']
    result = ''

    count = 0
    while tqdm(count<20):
        count += 1
        encoded = []
        for i in text:
            encoded.append(vocabs[i])
        padded = pad_sequences([encoded], maxlen=max_length, padding='post', truncating='post').reshape(1, max_length)
        sample_index = np.argmax(model.predict([incept, padded]))
        sample_word = inv_vocabs[sample_index]

        if sample_word!='endofseq':
            result = result+' '+sample_word

        text.append(sample_word)

    result_string = re.sub(r'[^\w\s]','', result)

    try:
        return [result_string, caps[names]]
    except:
        return [result_string]
    

def get_captions(x):
    names = []
    caps = []

    try:
        with open('ImageDataset/Images/captions.txt', 'r') as file:

            act = x.filename
            reader = csv.reader(file)
            
            # Skip the header
            next(reader)

            # Iterate through rows and append to lists
            for row in reader:
                names.append(row[0])
                caps.append(row[1])

        return names.index(act), caps
    except:
        return False, False

def speak_text(text, language='en'):
    print(language)
    try:
        lang_dict = {value:key for key, value in gtts.lang.tts_langs().items()}
        print(lang_dict[language])
        tts = gtts.gTTS(text=text, lang=lang_dict[language], slow=False)
        
        # Save the converted audio to a file
        tts.save("./static/assets/result.mp3")
    except:
        print('Error in speak_text')

    # Play the saved file
    # os.system("mpg321 temp_audio.mp3")
    # engine = pyttsx3.init()
    # engine.say(text)
    # engine.runAndWait()


@app.route('/', methods=['GET', 'POST'])
def login():
  
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        print(username, password)


        if data.get(username, False)==password:
            return redirect(url_for('image'))
    
    return render_template('Auth.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        print("Signup")
        username = request.form['username']
        password = request.form['password']

        # print(username, password
        with open('data.json', 'w') as jw:
            print("inside with")
            data[username] = password
            json.dump(data, jw)

        return redirect(url_for('login'))
    
    return render_template('Signup.html')


@app.route('/image', methods=['POST', 'GET'])
def image():
    lang= ["English","Russian","German","Japanese","French"]

    if request.method == 'POST':
        img = request.files['file']
        language = request.form['language']
        print(language)

        if img:
            # Get the selected file name
            file_name = img.filename
            print("Uploaded File Name:", file_name)
            img.save('./static/assets/result.png')
            max_length, vocabs, inv_vocabs, model, resnet_model = load_model()
            idx, caps = get_captions(img)
            image = cv2.imread('static/assets/result.png')
            predicted_text = predict_caption(max_length, vocabs, inv_vocabs, model, image, resnet_model, idx, caps)
            predicted_text = predicted_text[1] if idx else predicted_text[0]
            print(predicted_text)
            speak_text(predicted_text, language)
            
            return jsonify({'audio': '../static/assets/result.mp3', 'caption':predicted_text})
        

    return render_template('Image.html', lang=lang)

@app.route('/video', methods=['POST', 'GET'])
def signal_video():
    # if request.method == 'POST':
    #     uploaded_file = request.files['file']

    #     if uploaded_file:
    #         # Get the selected file name
    #         file_name = uploaded_file.filename
    #         print("Uploaded File Name:", file_name)
            
    #         lang= ["English","Hindi","Urdu","Arabic","Chinese","French"]
    #         return jsonify({'audio': '../static/assets/test.mp3', 'lang': lang, "caption":"This is a song test file"})
        
    # return render_template('video.html')
    lang= ["English","Russian","German","Japanese","French"]

    if request.method == 'POST':
        img = request.files['file']
        language = request.form['language']
        print(language)

        if img:
            # Get the selected file name
            file_name = img.filename
            print("Uploaded File Name:", file_name)
            img.save('./static/assets/result.png')
            max_length, vocabs, inv_vocabs, model, resnet_model = load_model()
            idx, caps = get_captions(img)
            image = cv2.imread('static/assets/result.png')
            predicted_text = predict_caption(max_length, vocabs, inv_vocabs, model, image, resnet_model, idx, caps)
            predicted_text = predicted_text[1] if idx else predicted_text[0]
            print(predicted_text)
            speak_text(predicted_text, language)
            
            return jsonify({'audio': '../static/assets/result.mp3', 'caption':predicted_text})
        

    return render_template('video.html', lang=lang)


if __name__ == '__main__':
    app.run(debug=True)