all:
	pip install poetry
	poetry install
	make surfnet-dev-local
	make surfnet-prod-local


surfnet-dev-local:
	export FLASK_ENV=development
	export FLASK_APP=src/plasticorigins/serving/app.py
	poetry run flask run

surfnet-prod-local:
	poetry run gunicorn -w 5 --threads 2 --bind 0.0.0.0:8000 plasticorigins.serving.wsgi:app
surfnet-prod-build-docker:
	poetry run docker build -t surfnet/surfnet:latest .

surfnet-prod-run-docker:
	poetry run docker run --env PYTHONPATH=/src/ -p 8000:8000 --name surfnetapi surfnet/surfnet:latest
