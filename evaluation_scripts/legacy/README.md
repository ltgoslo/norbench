# Norbench

### Description

At the moment, we have the evaluation scripts for [4 NLP tasks](http://wiki.nlpl.eu/Vectors/norlm/norbert). 

* In the current documentation, information about 3 of 4 tasks is provided in details:
  + [Fine-grained Sentiment Analysis task](#FINEGRAINED) -- detailed information about the current task is provided in the repository by link in the section
  + [Sentiment Analysis task](#sentiment-analysis-task)
    - [Data](#SENT_DATA)
    - [Evaluation](#SENT_EVAL)
    - [Models](#SENT_MODELS)
  + [Part-Of-Speech tagging task](#part-of-speech-tagging-task)
    - [Data](#POS_DATA)
    - [Evaluation](#POS_EVAL)
    - [Models](#POS_MODELS)
  + [Named Entity Recognition task](#named-entity-recognition-task)
    - [Data](#NER_DATA)
    - [Evaluation](#NER_EVAL)
    - [Models](#NER_MODELS)
* [Run all tasks](#run-all-tasks) 
   + [Parameters](#ALL_PARAMS) 
   + [Running scripts. Examples](#ALL_PARAMS_EXP) 

---

### <a name="FINEGRAINED"></a> Fine-grained Sentiment Analysis Task

The code and overall discription for the current task can be found [in this repo](https://github.com/jerbarnes/sentiment_graphs)

---

### Sentiment Analysis Task

#### Data

1. **Sentence-level.** 

    It's possible to finetune and evaluate 2 types of sentence-level SA: binary (negative and positive reviews only) and 3 classes (negative, neutral and positives). 

2. **Document-level.**

    While the reviews originally come with numerical ratings on a scale of 1–6, we here conflate these to three classes; negative (1–3), fair (4), and positive (5–6). This mapping is done to avoid problems with too few examples for the ratings in the extreme ends of the numerical scale. The dataset comes with predefined datasplits (chronologically sorted). More information about project you an fine on the website of [University of Oslo](https://www.mn.uio.no/ifi/english/research/projects/sant/).


Please specify **--task_specific_info** argument as required:
- **'sentence_2'** if you want to finetune/evaluate binary sentence-level SA.
- **'sentence_3'** if you want to finetune/evaluate 3 classes sentence-level SA.
- **'document_3'** if you want to finetune/evaluate 3 classes document-level SA.
- **'other'** if you want to finetune/evaluate on your own data and not NoReC datasets. In this case ```--data_path``` argument has to be specified.


#### Evaluation
Please specify `--data_path` argument only if you want to use your own datasets. In this case provide a path to folder with train.csv, dev.csv and test.csv. Datasets should contain columns 'review' with texts and 'sentiment' with labels. If you want to use sentence or document level datasets no need to specify this argument. Data downloads automatically (only once) from github repos and creates a folder with train, test and dev that are used in further experiments. Hence, when ```--task_specific_info``` is not 'other' datasets will be created automatically. 

There are additional arguments possible but not required for this task:
<ul>
  <li>--learning_rate</li>
  <li>--max_length</li>
  <li>--batch_size</li>
  <li>--epochs</li>
</ul>

#### Models

It's possible to use different architectures for fine-tuning/evaluating sentiment analysis:

- Masked language models
- Custom wrappers
- T5

If you want to use a custom wrapper please specify ```-custom_wrapper``` as True (it's False by default) and ```-path_to_model``` as absolute(!) path to folder with your model.
For any other model you can provide a path to HuggingFace model in ```-path_to_model```.

---

### Part of Speech Tagging Task


#### Data

To solve Part-of-speech tagging task, collected datasets for several languages can be used. Links to each of them with detailed information provided below:

* [Bokmaal](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal)
* [Nynorsk](https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk)
* [NynorskLIA](https://github.com/UniversalDependencies/UD_Norwegian-NynorskLIA)


#### <a name="POS_EVAL"></a>  Evaluation

Accuracy is used to evaluate this task. 
The calculation of the metric takes place inside the script, so the user receives a table with the accuracy scores obtained on the validation subset 
and on the testing data. The table with the output scores is automatically stored in the `results` directory.

There are additional arguments possible but not required for this task (they are set by default):
<ul>
  <li>--learning_rate</li>
  <li>--max_length</li>
  <li>--batch_size</li>
  <li>--eval_batch_size</li>
  <li>--epochs</li>
</ul>

#### Models

For the Part-of-speech tagging task, models of different architectures for fine-tuning/evaluating can be used:

- Masked language models
- T5

---

### Named Entity Recognition Task

#### Data

To detect named entities, datasets for several Norwegian languages were used: for Bokmaal and Nynorsk.

The classes provided below have been allocated for the token classification.
Each token is assigned to a tag according to the appropriate entity.
In the case when one entity marks several consecutive tokens, the first entity tag is assigned a token with the prefix B-, and the rest of the tokens for marked entity - with prefix I

* *O* - No entity
* *PER* - Person
* *LOC* - Location
* *PROD* - Product
* *GPE_LOC* -  Geo-political entity + locative sense
* *DRV* - Derived
* *EVT* - Event
* *ORG* - Organisation
* *GPE_ORG* - Geo-political entity + organisation sense
* *MISC* - Miscellaneous

You can find detailed information about the data and classes on the website of the [original project](https://github.com/ltgoslo/norne)

#### <a name="NER_EVAL"></a>  Evaluation

F1 score is used to evaluate this task.
The calculation of the metric takes place inside the script, so the user receives the table with the F1 score scores obtained on the testing data. 
The table with output scores is automatically stored in the `results` directory.

NOTE: for the current task not F1 score itself is used (not the original metric). 
A special [script](https://github.com/ltgoslo/norbench/evaluation_scripts/ner_eval.py) is used to get the result on the test set 
where scores are counted for provided labels: `PER`, `ORG`, `LOC`, `GPE_LOC`, `GPE_ORG`, `PROD`, `EVT`, `DRV`. 

There are additional arguments possible but not required for this task (they are set by default):
<ul>
  <li>--use_seqeval_evaluation</li>
  <li>--learning_rate</li>
  <li>--max_length</li>
  <li>--batch_size</li>
  <li>--eval_batch_size</li>
  <li>--epochs</li>
</ul>

#### Models

For the NER task, models of different architectures for fine-tuning/evaluating can be used:

- Masked language models
- T5

---

### Run all tasks
To run benchmark tasks `norbench_run.py` should be used.

The current script provides the ability to run all benchmark tasks (that were mentioned above in details) for one model or all models for one task.

#### <a name="ALL_PARAMS"></a>  Parameters

* `--task` - the name of the task: pos/ner/sentiment/all should be used. If nothing was entered by user, all tasks will be run

* `--task_specific_info` - Name of a sub-task. For origianl datasets name of language (e.g. Bokmaal \ Nynorsk \ NynorskLIA - for pos-tagging) or type of classification (sentence_2 / document_3) could be used.

* `--path_to_dataset` - path to the folder with data for current task. If 'ner' was chosen as a task, folder with the corresponding dataset should be entered. If one wants to run all benchmark tasks, there are other arguments that are more suitable for this (mentioned below). If path to the folder with data is empty, original datasets will be downloaded and used automatically.

* `--download_cur_data` - Current argument provides user to download the original datasets from github and use original dataset if it has been already downloaded: True if downloading is needed (False as default).

* `--path_to_dataset_pos` - path to the folder with data for pos task if 'all' in task was used. Should be used if user wants to run task on datasets other than original ones. If path to the folder with data is empty, original datasets will be downloaded and used automatically.

* `--path_to_dataset_ner` - path to the folder with data for ner task if 'all' in task was used. Should be used if user wants to run task on datasets other than original ones. If path to the folder with data is empty, original datasets will be downloaded and used automatically.

* `--path_to_dataset_sent` - path to the folder with data for binary sentiment task if 'all' in task was used. Should be used if user wants to run task on datasets other than original ones. If path to the folder with data is empty, original datasets will be downloaded and used automatically.

* `--model_name` - Name of the model that will be used as an identifier for checkpoints.

* `--path_to_model` - Path to model / 'all' to run all models that were tested for Norbench (can be added either as path or as full model name from Higgingface library). 

  Path to model can be entered in severeal ways:
  + Firstly: in model_names (`model_utils.py`), the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `mbert`
  + model name can be submitted as a full model name mentinoed in transformer library: `bert-base-multilingual-cased`
  + filesystem path to the model available locally

* `--do_train` - True if model will be trained from scratch. Train, evaluation and test will be run.

* `--custom_wrapper` -  a parameter that determines if custom wrapper should be specified (False by default). For the current moment this option is available for sentiment task (ref. Sentiment Analysis Task for more information).
* `--use_seqeval_evaluation_for_ner` -  boolean variable indicating whether to use the seqeval metric during validation
* `--epochs` -  number of training epochs (`10` by default)
* `--max_length` - max length for the texts
* `--batch_size` - size of batch for the training loop
* `--eval_batch_size` - size of batch for evaluation 
* `--learning_rate` - learning rate
    
 
A general launch option is provided below

```
  python3 ./norbench/evaluation_scripts/norbench_run.py
  
      --task {TASK_NAME}
      --task_specific_info {INFO_FORTHE_CURRENT_TASK}
      --model_name {MODEL_NAME_FOR_CHECKPOINTS}
      --path_to_model {PATH_TO_MODEL}
      --path_to_dataset {PATH_TO_FOLDER_WITH_DATA}
      --download_cur_data {DOWNLOAD_RELEVANT_DATA}
      --epochs {EPOCHS}
      --max_length {MAX_LENGTH}
      --batch_size {BATCH_SIZE}
      --eval_batch_size {EVAL_BATCH_SIZE}
      --learning_rate {LEARNING_RATE}
```


#### <a name="Overall_Run"></a> Running scripts. Examples

1. Run all tasks for a current model

    With the script provided below, it is possible to run all tasks with the original datasets (that will be downloaded)

    ```
    python3 ./norbench/evaluation_scripts/norbench_run.py --path_to_model scandibert --task all  --download_cur_data True
    ```

    if datasets were downloaded,it is possible not the mention `--download_cur_data` explicitly (but if the following argument will be mentioned, the original data will be used)

    ```
    python3 ./norbench/evaluation_scripts/norbench_run.py --path_to_model scandibert --task all
    ```

    to run all tasks with the datasets from the local folders:
    ```
    python3 ./norbench/evaluation_scripts/norbench_run.py --path_to_model norbert2 --task sentiment --path_to_dataset_pos {PATH_TO_FOLDER_POS} --path_to_dataset_ner {PATH_TO_FOLDER_NER} --path_to_dataset_sent {PATH_TO_FOLDER_SENTIMENT}
    ```

2. Run all models for a current task

    With the script provided below, it is possible to run sentiment task with the original datasets (that will be downloaded) for all NorBench models

    ```
    python3 ./norbench/evaluation_scripts/norbench_run.py --task sentiment --path_to_model all
    ```

3. Run a current model for a current task

    With the script provided below, it is possible to run sentiment task with the original datasets (that will be downloaded) for norbert2 model

    ```
    python3 ./norbench/evaluation_scripts/norbench_run.py --path_to_model norbert2 --task sentiment
    ```

    to run sentiment task with the dataset from the local folder:
    
    ```
    python3 ./norbench/evaluation_scripts/norbench_run.py --path_to_model norbert2 --task sentiment --path_to_dataset {PATH_TO_FOLDER}
    ```

4. Run all tasks with all the models

    With the script provided below, it is possible to run all tasks with the original datasets (that will be downloaded) for all model

    ```
    python3 ./norbench/evaluation_scripts/norbench_run.py --path_to_model all --task all
    ```
    

