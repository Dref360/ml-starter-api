ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

run:
	CFG_PATH='${ROOT_DIR}/config/conf.json' poetry run python runner.py --debug

test: mypy
	poetry run flake8 ml_starter_api
	poetry run pytest tests

format:
	poetry run black ml_starter_api

mypy:
	poetry run mypy ml_starter_api

compose:
	mkdir -p mongo-volume
	docker-compose up --build