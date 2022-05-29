from flask import Flask, render_template, request

from tweak import find_visuals
import tensorflow as tf

# Load inference model
model = tf.keras.models.load_model("../model")

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        # load data from the form sent back by client
        low_bound = float(request.form["lb"])
        up_bound = float(request.form["ub"])
        digit = int(request.form["digit"])
        dim = int(request.form["dim"])
        intervals = int(request.form["intervals"])
        if low_bound >= up_bound:
            message = "The upper bound should be larger than the lower bound."
            return render_template("index.html", message=message)

        # save images to `static` folder and return values for displaying.
        lower, upper, interval = find_visuals(model, digit, dim, low_bound, up_bound, intervals)

        return render_template("index.html", generated=True, lb=lower, ub=upper, interval=interval, digit=digit, dim=dim)
    return render_template("index.html", generted=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
