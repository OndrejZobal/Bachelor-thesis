Files:
    - anotate.py: Algorithm for processing diffs for training the patch-based error correction models
    - convert_bart_to_led.py: A script for using BART parameters to create a LED model
    - covert_to_peft.py: Adapt a model using QLoRA
    - dataset_formator.py: A library of input preprocessing functions that alter the dataset for training on different tasks
    - eval.py: A script for sampling the models, used together with measure.py to assess performance
    - extract-git-files.py: Extracts bug fix commits from a given repository
    - gpt-eval.py: A script for sampeling OpenAI API
    - heuristic.py: A library with the algorithm for predicting comment location
    - measure.py: A script for measuring performance metrics from eval.py's files
    - my_data.py: A library for loading datasets
    - my_params.py: A library for processing command line parameters
    - plbart_preprocess_fn.py: A script for converting code in a dataset to PLBART's format
    - preprocess_for_plbart.py: A library for processing code for PLBART
    - server.py: Serve requests for the CodeImprove client
    - train.py: Train an LED or a QLoRA-adapted model

    - eval_*.sh: These scripts are for running evaluations for their respective models
    - preprocess_*.sh: These scripts run dataset preprocessing for their respective models (note that docstrings and comments use the same script).
    - train_*.sh: These scripts are for training each respective model

    - config.toml: A configuration file for the CodeImprove server

    - CodeImprove Extension/: Contains the source files for the client extension.

    - requirements.txt: A list of required python packages

The workflow might look something like this:
    - Preprocessing the data (assuming codexglue_train.jsonl and codexglue_valid.jsonl exist and contain the dataset):
        bash preprocess_comments.sh codexglue_train.jsonl
        bash preprocess_comments.sh codexglue_valid.jsonl
    - Training the model:
        bash train_comment_generation.sh
    - Sampling outputs:
        bash eval_comment_generation.sh
    - Manually enable desired metric in measure.py
    - Measure performance:
        python3 measure.py report.jsonl
