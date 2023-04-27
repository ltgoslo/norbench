# Targeted Sentiment Analysis for Norwegian

This repository provides code for fine-tuning models on [Targeted Sentiment Analysis](https://github.com/ltgoslo/norec_tsa). It can laso be used for any other sequence labelling task. With "Sequence labelling" we speak of identifying a sequence in a text, consisting of one or more consequtive words, and to label that sequence correctly. Regarding modelling, it is the same task as Named Entity Recognition (NER), and this code can also be used to model the Norwegian NER data in [NorNE](https://github.com/ltgoslo/norne).
Note that Huggingface now calls this task [token classification](https://huggingface.co/docs/transformers/v4.28.1/en/tasks/token_classification). There is another task,  "text classification", where the task is, for one piece of text to output one label. Se the tutorial page [Text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification).  "sequence classification" has also been used as a synonym to "text classification"

We prefer to describe our task as "sequence labelling", the task which at Huggingface now is called "token classification"

# In this repository
- `helpers.ipynb` contains code that can be run before or after the training / fine-tuning, to prepare the data and to gather and analyze the results
- `create_fine_tune_json.py` is run before the training, to create config json files for the training. We use this to iterate alternative models and grid-search hyper-parameter tuning. This file needs inspection and modification according to the user's needs, in order to create the right output.

- `seq_label.py` is the main file for training a sequence labelling model. It takes one parameter, the path to a json configurations file. `seq_label.py` is derived from a [Huggingface example script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py). Our script has removed many options, and if you need those, use the original script in stead.
- `seq_label_testing.py` reads the json config file containing best epoch for that configuration, and also the numer of seeds to use in the test runs. Each model is fine-tuned and tested for the set amount of epochs, to find the mean and standard deviation on the test set.

- `run_sq_label_fox.slurm` and other slurm scripts are the slurm file we use to run our experiments. PyTorch with CUDA is taken care of in the module(s) loaded at the HPC
- `requirements_addon.txt` contain the extra required Python packagees to load. The versions of each package are not specified. This is done to avoid problems from specifying versions that do not play well witt other packages in the environment. Our versions have been: `Numpy: 1.20.3 , PyTorch: 1.11.0 , Transformers: 4.28.0`
- `modelling norbert.py`and `configuration_norbert.py` contain wrapper files that presently are required to fine-tune the NorBERT_3 models in a HF Trainer.


## Usage
[helpers.ipynb](helpers.ipynb) contain some helpful code, that can be run on your computer, before or after the model training.
- Convert the conll data enclosed into DatasetDict and save it as binary.
- A list of model names and paths that you may or may not find helpful
- Code for reading the outputs from experiments into a table

[create_fine_tune_json.py](create_fine_tune_json.py) is where you create the json file for each experiment. Make sure everything is set up according to your needs. Put the configuration files in subfolders as you find it useful.

[seq_label.py](seq_label.py) takes a config file as argument, fine-tunes the model accordingly and saves its results back to the same config file. 
Some notable differences:
    - Our script requires a json file woth configurations, while the HF script allows for individual parameters to be sent to the script in stead.
    - Our script requires the data to be in the Dataset binary (arrow) format.
    - Our script does not relate well to a model already fine-tuned, since that fine-tuning may have been on a different label to int mapping that what we create. The HF script has code for that situation.
    - When a word is split into subword tokens by the tokenizer, our script only relates to predicting the label for the first subword token. We have found from ewxperience that this is a good approach, and the other options were removed for readability of the script.
    - We save the `trainer.state.log_history` into the json config file. Information on the best epoch is also saved there. 
    - We use both [batista](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/) and [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) for evaluation. This is due to our particular interest in the matter.  The log history records seqeval evaluations, while the final testing records the batista scores. The seqeval and batista values are always very close, but not always completely the same.

[seq_label_testing.py](seq_label_testing.py) takes a config file as argument, where best epoch is added by `seq_label.py`. For each seed created, a model is fine-tuned (again) for the beat amount of epochs, and the results are stored back to the original json config file.

**slurm-files** are used for iterating the experiment config files. 

## Requirements
 `run_sq_label_fox.slurm` shows how the required environment is created. Comment out cleaning and reinstalling the environment when this is not needed. When running on other infrastructure, intall a suitable Python and PyTorch version. On Fox, I am getting these versions:
 ```
 Numpy: 1.20.3
PyTorch: 1.11.0
Transformers: 4.28.0
``` 

