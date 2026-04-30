from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model safely
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])

        if model is None:
            return "Model not loaded properly"

        prediction = model.predict([[f1, f2, f3, f4]])

        return f"Prediction: {prediction[0]}"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
