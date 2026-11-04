.PHONY: install test analyse fetch clean all

install:
	pip install -r requirements.txt

test:
	python -m pytest tests -q

analyse:
	python -m src.pmcal.cli --data data/raw/markets.csv --out results --figures figures

fetch:
	python scripts/fetch_polymarket.py --out data/raw/markets.csv

all: test analyse

clean:
	rm -rf results/*.json results/*.md figures/*.png .pytest_cache
