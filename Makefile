.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

run:
	python src/full_pipeline.py

start_app:
	poetry run uvicorn app:app --reload

build_docker_image:
	docker build . -t flood-prediction-api:latest
