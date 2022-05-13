import logging
import logging.config

from flask import Flask, render_template, request

from plasticorigins.serving.config import logging_config
from plasticorigins.serving.inference import handle_post_request

logging.config.dictConfig(logging_config)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return handle_post_request()
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(threaded=True, port=5000, debug=False, host="0.0.0.0")
