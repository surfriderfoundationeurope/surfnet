export FLASK_ENV=development
export FLASK_APP=src/plasticorigins/serving/app.py

install:

	pip install poetry
	poetry install


surfnet-dev-local: install

	poetry run flask run

surfnet-prod-local: install
	poetry run gunicorn -w 5 --threads 2 --bind 0.0.0.0:8000 plasticorigins.serving.wsgi:app
surfnet-prod-build-docker:
	poetry run docker build -t surfnet/surfnet:latest .

surfnet-prod-run-docker: install
	poetry run docker run --env PYTHONPATH=/src/ -p 8000:8000 --name surfnetapi surfnet/surfnet:latest
