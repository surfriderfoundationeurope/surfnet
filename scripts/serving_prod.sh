#/bin/bash

gunicorn -w 5 --threads 2 --bind 0.0.0.0:8000 --chdir /src/serving/ wsgi:app