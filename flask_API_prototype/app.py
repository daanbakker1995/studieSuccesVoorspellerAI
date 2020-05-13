from flask import Flask
from flask import render_template

# Initialize a Flask app.
app = Flask(__name__)


# GET - /
@app.route('/')
def index():
    # Return templates/index.html.
    # This only works if the
    return render_template('index.html')


# POST - /api/v1/prediction
@app.route('/api/v1/prediction', methods=['POST'])
def create_task():
    return '{}'


# Run the app in debug mode if the Python interpreter runs
# the current file as its main program.
if __name__ == '__main__':
    app.run(debug=True)
