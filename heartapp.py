import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("heartmodel.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("heart.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction[0] == 0:
        prediction_text = "The person does not have a heart problem."
    else:
        prediction_text = "The person has a heart problem."
    return render_template("heart.html", prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True)