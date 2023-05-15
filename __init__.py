from flask import Flask, render_template
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flaskext.markdown import Markdown
from sqlalchemy import MetaData

import config


##config로 RDS랑 연결하기
##오류 페이지를 프론트에서 이쁘게 만들어준다면 errorHandler만들기

db = SQLAlchemy()
migrate=Migrate()

                
def create_app():
    
    app = Flask(__name__)
    app.config.from_envvar('APP_CONFIG_FILE')

    # ORM
    db.init_app(app)
    app.config.from_pyfile("config.py")

    from . import models

    # controller
    from .views import *
    app.restister_blueprint(main_views.bp)

    # filter                                     
    from .filter import format_datetime


    # markdown
    Markdown(app, extensions=['nl2br', 'fenced_code'])
    
    database_connection = db.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'
            .format(user, password,host, dbname)).connect()

    return app