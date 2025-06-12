.PHONY: build_image

build_image:
	docker build --tag=nlp_project .

run_container:
	docker run -d -it nlp_project --name nlp_project

attach_container:
	docker exec -it nlp_project bash

format_code:
	ruff format .

lint_code:
	ruff check .
