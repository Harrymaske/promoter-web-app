from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("promoter_rf_model.pkl")

def encode_sequence(seq):
    base_dict = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'C': [0, 0, 0, 1]
    }
    seq = seq.upper()
    encoded = []
    for base in seq:
        encoded.extend(base_dict.get(base, [0, 0, 0, 0]))
    return encoded

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        sequence = request.form["sequence"]
        if len(sequence) == 57:
            encoded = encode_sequence(sequence)
            pred = model.predict([encoded])[0]
            proba = model.predict_proba([encoded])[0][1]
            prediction = "Promoter" if pred == 1 else "Not a promoter"
            confidence = round(proba if pred == 1 else 1 - proba, 2)
        else:
            prediction = "❌ Error: Sequence must be 57 bases long."

    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
