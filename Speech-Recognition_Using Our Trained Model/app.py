from flask import Flask, render_template, request, redirect
from tensorflow.keras.applications.vgg16 import preprocess_input,VGG16
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
import speech_recognition as gsr
import numpy as np
import matplotlib.pyplot as plt
import librosa

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    text_google=""
    text_vgg=""
    transcript_vgg = ""
    transcript_google = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # VGG
            y, sr = librosa.load(file, sr=16000)

            # Compute the Mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            
            # Convert power spectrogram to dB-scale mel spectrogram
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Save the Spectrogram
            fig = plt.figure(figsize=[1,1])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            librosa.display.specshow(S_dB, cmap='gray_r')
            plt.savefig('spectrogram.png', dpi=400, bbox_inches='tight',pad_inches=0)
            plt.close()

            # Image Loading and Feature Extracting
            image = load_img('spectrogram.png', target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)

            feature_model = VGG16()
            feature_model = Model(inputs = feature_model.inputs , outputs = feature_model.layers[-2].output)
            feature = feature_model.predict(image, verbose=0)
            print(feature)

            # Preprocessing features
            with open('captions.txt', 'r') as f:
                captions = f.readlines()

            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(captions)
            vocab_size = len(tokenizer.word_index) + 1
            max_length = 25

            def idx_to_word(integer, tokenizer):
                for word, index in tokenizer.word_index.items():
                    if index == integer:
                        return word
                return None

            def predict_caption(model, image, tokenizer, max_length):
                # add start tag for generation process
                in_text = 'startseq'
                # iterate over the max length of sequence
                for i in range(max_length):
                    # encode input sequence
                    sequence = tokenizer.texts_to_sequences([in_text])[0]
                    # pad the sequence
                    sequence = pad_sequences([sequence], max_length)
                    # predict next word
                    yhat = model.predict([image, sequence], verbose=0)
                    # get index with high probability
                    yhat = np.argmax(yhat)
                    # convert index to word
                    word = idx_to_word(yhat, tokenizer)
                    # stop if word not found
                    if word is None:
                        break
                    # append word as input for generating next word
                    in_text += " " + word
                    # stop if we reach end tag
                    if word == 'endseq':
                        break
                return in_text

            best_model = load_model('best_model.h5',compile=False)
            prediction = predict_caption(best_model, feature, tokenizer, max_length)
            words = prediction.split()
            new_words = [word for word in words if word not in ['startseq', 'endseq']]
            text_vgg = ' '.join(new_words)
            print(text_vgg)
            a={"transcript":text_vgg}
    return render_template('main.html', transcript=text_vgg)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)