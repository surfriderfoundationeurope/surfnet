#/bin/bash

poetry run gunicorn -w 5 --threads 2 --bind 0.0.0.0:8000 plasticorigins.serving.wsgi:app