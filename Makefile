.PHONY: all update analyse status install help

PYTHON := python3
MAIN   := $(PYTHON) main.py

## all      : fetch any missing data, run analysis, and save charts (default)
all:
	$(MAIN)

## update   : fetch missing price and weather data into the DB only
update:
	$(MAIN) update

## analyse  : run correlations, regression, predictions and save charts
analyse:
	$(MAIN) analyse

## status   : show date ranges currently stored in the DB
status:
	$(MAIN) status

## install  : install Python dependencies
install:
	pip3 install --break-system-packages requests pandas numpy matplotlib scikit-learn scipy plotly yfinance

## help     : list available targets
help:
	@grep "^##" Makefile | sed 's/^## /  /'
