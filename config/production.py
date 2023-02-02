from logging.config import dictConfig

from config.default import *

SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(
    user="admin",
    pw="ringmybell",
    url="ringmybell2.cpuun9hfqr17.ap-northeast-2.rds.amazonaws.com:3306/ringmybell",
    db="ringmybell")
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = b'6rD+XqRaLrmik+xHCc15O8Rm8tNi/gSIAhDeOqM6'

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs/myproject.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5 MB
            'backupCount': 5,
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file']
    }
})