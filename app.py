from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))


@app.route('/')
def home():

    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form.get("content")
    email_tokenized = tokenizer.transform([email])
    predictions = model.predict(email_tokenized)
    predictions = 1 if predictions == 1 else -1
    return render_template("index.html", predictions=predictions, email_text=email)


if __name__ == "__main__":
    app.run(debug=True)
