from flask import Flask
from routes import app as routes_app
from api import api_app as api_app

app = Flask(__name__)

# Регистрация маршрутов
app.register_blueprint(routes_app)
app.register_blueprint(api_app, url_prefix='/api')

if __name__ == "__main__":
    app.run(debug=True)