from flask import Flask
from routes import app as routes_app
from api import api_app

app = Flask(__name__)

# Настройки для загрузки файлов
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Регистрация маршрутов
app.register_blueprint(routes_app)
app.register_blueprint(api_app, url_prefix='/api')

if __name__ == "__main__":
    app.run(debug=True)