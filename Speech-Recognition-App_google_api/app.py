from flask import Flask, render_template, request, redirect
import speech_recognition as sr

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    text=""
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
    
            # Load the Assamese audio file
            
            
            
            # Use Google Speech-to-Text API to recognize the speech
            try:
                text = recognizer.recognize_google(data, language="as-IN")  # Specify Assamese (as-IN) as the source language
                print(text)
            except sr.UnknownValueError:
                print("Unable to recognize speech")
            except sr.RequestError as e:
                print(f"Error: {e}")

    return render_template('main.html', transcript=text)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
