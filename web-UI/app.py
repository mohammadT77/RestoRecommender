from flask import Flask, request, render_template, session
from flask_session import Session
from main import generate_response

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    # session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = generate_response(user_input)
    session['chat_history'].append(('user', user_input))
    session['chat_history'].append(('bot', response))
    return render_template('index.html', chat_history=session['chat_history'])

if __name__ == '__main__':
    app.run(debug=True)
