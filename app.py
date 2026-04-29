from flask import Flask, render_template, request
import pickle
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        f1 = float(request.form["f1"])
        f2 = float(request.form["f2"])
        f3 = float(request.form["f3"])
        f4 = float(request.form["f4"])

        features = np.array([[f1, f2, f3, f4]])
        prediction = model.predict(features)[0]

        if prediction == 0:
            result = "Healthy"
        else:
            result = "Fault Detected"

        # chart
        fig, ax = plt.subplots()
        names = ["F1", "F2", "F3", "F4"]
        values = [f1, f2, f3, f4]
        ax.bar(names, values)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        graph_url = base64.b64encode(img.getvalue()).decode()

        return render_template("index.html", result=result, graph=graph_url)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)