.PHONY: install test api preprocess train evaluate docker-build docker-run clean

install:
	pip install -r requirements.txt

test:
	pytest -q

api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

preprocess:
	python src/preprocess.py

train:
	python src/train_baseline.py

evaluate:
	python src/evaluate.py

docker-build:
	docker build -t ai-newsops-api .

docker-run:
	docker run --rm -p 8000:8000 ai-newsops-api

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete