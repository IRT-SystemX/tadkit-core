DOCKER_NETWORK=demo_tadkit
.PHONY: build run clear deep-clean


check_network:
	@if [ -z $(shell docker network ls -q -f name=$(DOCKER_NETWORK)) ]; then docker network create --driver=bridge --subnet=172.22.0.0/16 $(DOCKER_NETWORK); fi


build:
	docker build -t streamlit --network=host -f Dockerfile .
run:
	docker run -d -p 8501:8501 --network $(DOCKER_NETWORK) --name visualiser_app streamlit
enter:
	docker exec -it visualiser_app bash

clear:
	docker rm -f visualiser_app || true
	docker image rm streamlit || true

deep-clean:
	docker container prune -f
	docker image prune -a -f
	docker volume prune -f


###############################################################################

all: clear check_network build run
