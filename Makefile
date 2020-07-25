# Allennlp

ALLENNLP_PARAM_PATH := config/demo.jsonnet
ALLENNLP_SERIALIZATION_DIR := results_cli 

train-via-script:
	python run.py

train-cli:
	allennlp train ${ALLENNLP_PARAM_PATH} -s ${ALLENNLP_SERIALIZATION_DIR} --include-package many_tokenizer_as_one_model

test:
	echo "not yet"

tokenize:
	echo "not yet"
