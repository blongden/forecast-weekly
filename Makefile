.PHONY: all update analyse status install help \
        build pod-all pod-update pod-analyse pod-status pod-shell \
        cdk-install cdk-synth cdk-deploy cdk-destroy

PYTHON := .venv/bin/python3
MAIN   := $(PYTHON) main.py

IMAGE  := energy-analysis
# Mount the local DB and outputs into the container so data persists
POD    := podman run --rm \
            -v $(PWD)/energy.db:/data/energy.db \
            -v $(PWD)/charts:/data/charts \
            -v $(PWD)/index.html:/data/index.html \
            $(IMAGE)

## ── Local (direct) ────────────────────────────────────────────────────────────

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

## install  : create virtualenv and install Python dependencies
install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

## ── Container (Podman) ────────────────────────────────────────────────────────

## build        : build the container image
build:
	podman build -t $(IMAGE) .

## pod-all      : run full update + analyse cycle inside the container
pod-all: build
	$(POD) python main.py all

## pod-update   : fetch data only, inside the container
pod-update: build
	$(POD) python main.py update

## pod-analyse  : run analysis only, inside the container
pod-analyse: build
	$(POD) python main.py analyse

## pod-status   : show DB status from inside the container
pod-status: build
	$(POD) python main.py status

## pod-shell    : open a shell inside the container (useful for debugging)
pod-shell: build
	podman run --rm -it \
	  -v $(PWD)/energy.db:/data/energy.db \
	  -v $(PWD)/charts:/data/charts \
	  -v $(PWD)/index.html:/data/index.html \
	  $(IMAGE) /bin/bash

## ── AWS (CDK) ────────────────────────────────────────────────────────────────

## cdk-install  : install CDK Python dependencies
cdk-install:
	cd infra && pip install -r requirements.txt

## cdk-synth    : synthesise CloudFormation template (dry run)
cdk-synth:
	cd infra && CDK_DOCKER=podman cdk synth

## cdk-deploy   : deploy all AWS infrastructure
cdk-deploy:
	cd infra && CDK_DOCKER=podman cdk deploy --require-approval broadening

## cdk-destroy  : tear down all AWS infrastructure
cdk-destroy:
	cd infra && CDK_DOCKER=podman cdk destroy

## help         : list available targets
help:
	@grep "^##" Makefile | sed 's/^## /  /'
