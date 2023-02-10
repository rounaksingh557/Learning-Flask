from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index() -> str:
    'This is the initial function of the app.'
    return render_template('index.html', name="Rounak Singh")


if __name__ == "__main__":
    app.run(debug=True)
