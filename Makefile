.PHONY: clean data lint requirements train

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = covidecg

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	python3 -m pip install -U pip setuptools wheel
	python3 -m pip install -r requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 covidecg

## Test python environment is setup correctly
test_environment:
	python3 test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


ecg2img:
	rm -rf t-dir data/processed/ecg2img_postcovid
	mkdir data/processed/ecg2img_postcovid
	python3 create_images.py --img-height 200 \
		--recordings-file data/interim/recordings_stress_ecg_postcovid.csv \
		--recordings-dir data/interim/recordings \
		--output-dir data/processed/ecg2img_postcovid

	rm -rf data/processed/ecg2img_ctrl
	mkdir data/processed/ecg2img_ctrl
	python3 create_images.py --img-height 200 \
		--recordings-file data/interim/recordings_stress_ecg_ctrl.csv \
		--recordings-dir data/interim/recordings \
		--output-dir data/processed/ecg2img_ctrl


## Make Dataset
data:
	
#	extract all recordings for postcovid patient group
	rm -rf data/interim/recordings_postcovid
	mkdir data/interim/recordings_postcovid
	python3 covidecg/data/extract_recordings.py \
		--prefix postcovid \
		--patients-list data/raw/patients_postcovid.csv \
		data/raw/ecg_export_postcovid data/interim/recordings_postcovid
	
#	extract all recordings for control group
	rm -rf data/interim/recordings_ctrl
	mkdir data/interim/recordings_ctrl
	python3 covidecg/data/extract_recordings.py \
		--prefix ctrl \
		--patients-list data/raw/patients_ctrl.csv \
		data/raw/ecg_export_ctrl data/interim/recordings_ctrl


#	concatenate recordings of the same session together
	python3 covidecg/data/concat_recordings_per_session.py \
		--recordings-list data/interim/recordings_stress_ecg_postcovid.csv \
		data/interim/recordings_postcovid data/interim/fullsessions_postcovid

	python3 covidecg/data/concat_recordings_per_session.py \
		--recordings-list data/interim/recordings_stress_ecg_ctrl.csv \
		data/interim/recordings_ctrl data/interim/fullsessions_ctrl


#	merge ecg recordings directories
	rm -rf data/interim/recordings
	mkdir data/interim/recordings
	ln -s $(shell pwd)/data/interim/recordings_postcovid/* $(shell pwd)/data/interim/recordings/
	ln -s $(shell pwd)/data/interim/recordings_ctrl/* $(shell pwd)/data/interim/recordings/

#	merge ecg recordings directories
	rm -rf data/interim/fullsessions
	mkdir data/interim/fullsessions
	ln -s $(shell pwd)/data/interim/fullsessions_postcovid/* $(shell pwd)/data/interim/fullsessions/
	ln -s $(shell pwd)/data/interim/fullsessions_ctrl/* $(shell pwd)/data/interim/fullsessions/




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
