ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DEVICE=cpu

ifeq ($(DEVICE),gpu)
	COMPOSE_EXT=-f docker-compose-gpu.yml
else
	COMPOSE_EXT=
endif

run:
	CFG_PATH='${ROOT_DIR}/config/conf.json' poetry run python runner.py --debug

test: mypy
	poetry run flake8 ml_starter_api
	poetry run pytest tests

format:
	poetry run black ml_starter_api

mypy:
	poetry run mypy ml_starter_api

build:
	docker build --build-arg DEVICE=$(DEVICE) \
				-t ml_starter_api/ml_starter_api_$(DEVICE) .

compose: build
	DEVICE=$(DEVICE) docker-compose -f docker-compose.yml $(COMPOSE_EXT) up