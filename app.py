from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Variable to track chatbot status
chatbot_active = True

@app.route('/')
def index():
    return render_template('index.html', chatbot_active=chatbot_active)

@app.route('/toggle', methods=['POST'])
def toggle():
    global chatbot_active
    chatbot_active = not chatbot_active
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
