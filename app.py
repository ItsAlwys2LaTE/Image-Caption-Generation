import os
import pickle
from flask import Flask, request, render_template, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----- CONFIG -----
UPLOAD_FOLDER = 'static/uploads'
MAX_LENGTH = 34          # set to your max caption length from training
START_TOKEN = 'startseq' # whatever you used in training
END_TOKEN = 'endseq'

# ----- CREATE APP -----
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----- LOAD MODEL & TOKENIZER -----
caption_model = tf.keras.models.load_model('caption_model.keras')  # saved in step 1[web:47]
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# If you used a separate CNN feature extractor (e.g. VGG16/InceptionV3) in training,
# load it here with the same architecture and preprocessing.

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

cnn_model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')  # 1920-dim

# ----- HELPER: EXTRACT IMAGE FEATURES -----
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = cnn_model.predict(img)
    return features  # shape (1, 4096)

# ----- HELPER: ID <-> WORD -----
index_word = {v: k for k, v in tokenizer.word_index.items()}

# ----- HELPER: GENERATE CAPTION (GREEDY) -----
def generate_caption(image_path):
    photo = extract_features(image_path)  # (1, feature_dim)

    in_text = START_TOKEN
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)

        yhat = caption_model.predict([photo, sequence], verbose=0)  # shape (1, vocab_size)
        yhat = np.argmax(yhat)

        word = index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == END_TOKEN:
            break

    # clean start/end tokens
    result = in_text.replace(START_TOKEN, '').replace(END_TOKEN, '').strip()
    return result

# ----- ROUTES -----
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', caption="No file part")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', caption="No selected file")

    # Save upload
    filename = file.filename  # you can wrap this with secure_filename later
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Generate caption
    caption = generate_caption(filepath)

    image_url = url_for('static', filename=f'uploads/{file.filename}')
    return render_template('index.html', image_url=image_url, caption=caption, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
