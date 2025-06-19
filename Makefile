.PHONY: build_image run_container attach_container format_code lint_code

build_image:
	docker build --tag=nlp_project .


run_container:
	docker run -d -it nlp_project --gpus=all -v ${PWD}:/workspace --name nlp_project


attach_container:
	docker exec -it nlp_project bash


format_code:
	ruff format .


lint_code:
	ruff check .
