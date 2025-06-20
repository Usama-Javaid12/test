from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Hello Azure! Deployment Test Successful ✅</h1><p>This is a minimal test app to verify Azure deployment is working.</p>'

@app.route('/health')
def health():
    return {'status': 'healthy', 'message': 'App is running successfully'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
