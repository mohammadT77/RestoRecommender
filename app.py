from flask import Flask, request, render_template
from main import generate_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = generate_response(user_input)
    return render_template('index.html', user_input=user_input, response=response)

if __name__ == '__main__':
    app.run(debug=True)
