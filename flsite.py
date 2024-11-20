from flask import Flask
from routes import routes_app
from api import api_app

def create_app():
    app = Flask(__name__)

    # Настройки для загрузки файлов
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

    # Создание папки uploads, если она не существует
    import os
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Регистрация маршрутов
    app.register_blueprint(routes_app)
    app.register_blueprint(api_app, url_prefix='/api')

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)