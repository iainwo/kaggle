.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = wids_datathon_2020
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m ipykernel install --user --name wids_datathon_2020 --display-name="wids_datathon_2020"

## Install Development Deps
requirements_dev: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.dev.txt

## Build python package
build: requirements
	$(PYTHON_INTERPRETER) -m pip install --user --upgrade setuptools wheel
	$(PYTHON_INTERPRETER) setup.py sdist bdist_wheel

build_app:
	docker image build -t iainwo/wids-datathon-2020 -f streamlit/Dockerfile.streamlit .

## Publish python package to PyPi
publish:
	$(PYTHON_INTERPRETER) -m pip install --user --upgrade twine
	$(PYTHON_INTERPRETER) -m twine upload dist/*

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) wids_datathon_2020/data/unzip_dataset.py data/external/widsdatathon2020.zip data/raw/
	$(PYTHON_INTERPRETER) wids_datathon_2020/data/make_dataset.py data/raw/training_v2.csv data/interim/
	$(PYTHON_INTERPRETER) wids_datathon_2020/data/stratify_dataset.py data/interim/training_v2.feather data/interim/
	$(PYTHON_INTERPRETER) wids_datathon_2020/data/encode_dataset.py --is-create-encoders data/interim/training_v2_train.feather data/processed/ models/
	$(PYTHON_INTERPRETER) wids_datathon_2020/data/encode_dataset.py data/interim/training_v2_val.feather data/processed/ models/
	$(PYTHON_INTERPRETER) wids_datathon_2020/data/encode_dataset.py data/interim/training_v2_test.feather data/processed/ models/
	$(PYTHON_INTERPRETER) wids_datathon_2020/data/make_dataset.py data/raw/unlabeled.csv data/interim/
	$(PYTHON_INTERPRETER) wids_datathon_2020/data/encode_dataset.py data/interim/unlabeled.feather data/processed/ models/
	# $(PYTHON_INTERPRETER) wids_datathon_2020/features/build_features.py data/interim data/processed

## Make Model
model: requirements
	$(PYTHON_INTERPRETER) wids_datathon_2020/models/train_model.py data/processed/training_v2_train_encoded.feather data/processed/training_v2_val_encoded.feather data/processed/training_v2_test_encoded.feather models/ reports/ reports/figures/

## Make Predictions
predictions: requirements
	$(PYTHON_INTERPRETER) wids_datathon_2020/models/predict_model.py data/processed/unlabeled_encoded.feather models/model.dump data/predictions

## Make Streamlit
app: build
	docker run -p8080:8080 iainwo/wids-datathon-2020:latest

## Generate visuals for feature EDA
eda: requirements
	$(PYTHON_INTERPRETER) wids_datathon_2020/visualization/visualize.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf dist/
	rm -rf catboost_info/
	rm -rf test-results.xml

## Lint using flake8
lint:
	flake8 wids_datathon_2020

## Run unit tests
test:
	py.test -v --junitxml="test-results.xml"

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
