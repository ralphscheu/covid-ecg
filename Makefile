.PHONY: clean data lint requirements features-medical features-lfcc features

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = covid-ecg
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

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 covidecg

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


## Make Dataset
data: 
	# Extract runs
	rm -rf data/interim/recordings_covid
	mkdir data/interim/recordings_covid
	$(PYTHON_INTERPRETER) covidecg/data/extract_recordings.py \
		--prefix covid \
		--patients-list data/raw/patients_covid.csv \
		data/raw/ecg_export_covid data/interim/recordings_covid
	
	rm -rf data/interim/recordings_postcovid
	mkdir data/interim/recordings_postcovid
	$(PYTHON_INTERPRETER) covidecg/data/extract_recordings.py \
	--prefix postcovid \
	--patients-list data/raw/patients_postcovid.csv \
	data/raw/ecg_export_postcovid data/interim/recordings_postcovid
	
	rm -rf data/interim/recordings_ctrl
	mkdir data/interim/recordings_ctrl
	$(PYTHON_INTERPRETER) covidecg/data/extract_recordings.py \
	--prefix ctrl \
	--patients-list data/raw/patients_ctrl.csv \
	data/raw/ecg_export_ctrl data/interim/recordings_ctrl

	# merge ecg run directories
	rm -rf data/interim/recordings
	mkdir data/interim/recordings
	ln -s $(shell pwd)/data/interim/recordings_covid/* $(shell pwd)/data/interim/recordings/
	ln -s $(shell pwd)/data/interim/recordings_postcovid/* $(shell pwd)/data/interim/recordings/
	ln -s $(shell pwd)/data/interim/recordings_ctrl/* $(shell pwd)/data/interim/recordings/

	# merge ecg run info csv files
	cp data/interim/recordings_covid.csv data/interim/recordings.csv
	tail -n +2 data/interim/recordings_postcovid.csv >> data/interim/recordings.csv
	tail -n +2 data/interim/recordings_ctrl.csv >> data/interim/recordings.csv


## Generate medical measurement features (Peaks, Intervals)
lfcc:
	# MFCC features over complete runs
	# ${PYTHON_INTERPRETER} covidecg/features/make_mfcc.py data/interim/recordings_covid data/interim/ecg_runs_mfcc_covid
	# ${PYTHON_INTERPRETER} covidecg/features/make_mfcc.py data/interim/recordings_ctrl data/interim/ecg_runs_mfcc_ctrl


## Generate features
features: features-medical features-lfcc

train_mlp:
	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/covid_ctrl_classification/01-covid_ctrl-recordings-plain_signal-mlp.yaml


train_cnn2d:
	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/covid_postcovid-recordings-plain_signal-cnn2d.yaml


train_cnn1d__raw_signal:
	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/covid_postcovid-recordings-plain_signal-cnn1d.yaml
#	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/postcovid_ctrl-recordings-plain_signal-cnn1d.yaml

train_cnn1d__lfcc:
	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/covid_postcovid-recordings-lfcc-cnn1d.yaml
#	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/postcovid_ctrl-recordings-lfcc-cnn1d.yaml

train_cnn1d: train_cnn1d__raw_signal train_cnn1d__lfcc


train: train_mlp train_cnn2d train_cnn1d
#	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/01-covid_ctrl-recordings-plain_signal-svmlinear.yaml
#	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/01-covid_ctrl-recordings-plain_signal-mlp.yaml

#	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/02-covid_ctrl-recordings-peaks-svmlinear.yaml

#	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/03-covid_ctrl-recordings-intervals-svmlinear.yaml
#	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/03-covid_ctrl-recordings-intervals-mlp.yaml

#	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/04-covid_ctrl-recordings-peaks_intervals-svmlinear.yaml
#	${PYTHON_INTERPRETER} ./train_evaluate.py --config-file ./exp_configs/04-covid_ctrl-recordings-peaks_intervals-mlp.yaml


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
