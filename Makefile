surfnet-dev-local:
	FLASK_ENV=development FLASK_APP=./src/serving/app.py flask run

surfnet-prod-local:
	PYTHONPATH=./src gunicorn -w 5 --threads 2 --bind 0.0.0.0:8001 --chdir ./src/serving/ wsgi:app

surfnet-prod-build-docker:
	docker build -t surfnet/surfnet:latest .

surfnet-prod-run-docker:
	docker run --env PYTHONPATH=/src -p 8000:8000 --name surfnetapi surfnet/surfnet:latest
