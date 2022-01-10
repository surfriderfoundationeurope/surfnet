from flask import Flask, render_template, request
import logging, logging.config
from serving.inference import handle_post_request

logging.config.dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }, 'file': {
        'class': 'logging.FileHandler',
        'formatter': 'default',
        'filename': 'errors.log'
    }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi','file']
    }
})

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return handle_post_request()
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(threaded=True, port=5000, debug=False, host="0.0.0.0")
