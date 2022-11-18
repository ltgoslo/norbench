# Norbench

### Description

At the moment, we have the evaluation scripts for [4 NLP tasks](http://wiki.nlpl.eu/Vectors/norlm/norbert). 

* [How repository is organized](#STRUCT) 
* In the current documentation information about 3 of 4 tasks is provided in details:
  + [Part-Of-Speech tagging task](#POS)
    - [Parameters](#POS_PARAMS)
    - [How to run the training script](#POS_SCRIPT)
    - [Evaluation](#POS_EVAL)
    - [Available Models](#POS_MODELS)
  + [Fine-grained Sentiment Analysis task](#FINEGRAINED) -- detailed information about the current task is provided in the repository by link in the section
  + [Binary Sentiment Analysis task](#BINARYSENT)
    - [Parameters](#BINARYSENT_PARAMS)
    - [How to run the training script](#BINARYSENT_SCRIPT)
    - [Evaluation](#BINARYSENT_EVAL)
    - [Available Models](#BINARYSENT_MODELS)
  + [Named Entity Recognition task](#NER)
    - [Parameters](#NER_PARAMS)
    - [How to run the training script](#NER_SCRIPT)
    - [Evaluation](#NER_EVAL)
    - [Available Models](#NER_MODELS)

### <a name="STRUCT"></a> Structure 

The repository contains the evaluation scripts. One should create the `data` 
directory with the corresponding datasets (training, vaidation, and test 
splits).

The structure of `data` should be the following:

```
--- data
  |
   --- pos
      |
       --- nob
           ...
       --- nno
           ...
  
   --- sentiment
       |
        --- no
            ...
   --- ner
       |
        --- nob
            ...
        --- nno
            ...
    
```

---

### <a name="POS"></a> Part of Speech Tagging Task


For this task, [script](https://github.com/ltgoslo/norbench/evaluation_scripts/pos_finetuning.py) `pos_finetuning.py` should be used.



#### <a name="POS_PARAMS"></a>  Parameters

The input of the model is:

* `--model_name` - the name of the model with which the user intends to save the output file

* `--short_model_name` - the name of the model can be presented in several ways.
  +  First: in get_full_model_names, the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `mbert`
  + or the model name can be submitted as a full model name mentioned in the Transformers library: `bert-base-multilingual-cased`
  + ...or simply as the filesystem path to the model available locally

* `--training_language` - as the POS-tagging task exists for both Norwegian Bokmål and Norwegian Nynorsk,  `nob` or `nno` should be used respectively

* `--epochs` - number of training epochs (`10` by default)


#### <a name="POS_SCRIPT"></a> Running script


Scripts could be run on the [SAGA](https://documentation.sigma2.no/index.html) cluster

In order to run the script on Saga, it is necessary to put arguments for [parameters](#POS_PARAMS) in the form indicated below.


Sample `.slurm` file could be found in [this](https://github.com/ltgoslo/norbench/evaluation_scripts/evaluate_pos_tagging.slurm) directory

```
python3 pos_finetuning.py --short_model_name ${MODEL} --training_language ${LANG} --model_name ${IDENTIFIER} --epochs 10

```

#### <a name="POS_EVAL"></a>  Evaluation

Accuracy is used to evaluate this task. 
The calculation of the metric takes place inside the script, so the user receives a table with the accuracy scores obtained on the validation subset 
and on the testing data. The table with the output scores is automatically stored in the `results` directory.

#### <a name="POS_MODELS"></a>  Models that have been successfully tested on this script

Currently, this script can work with Bert-Like-Models, DistilBert, and Xlm-Roberta models. and models which are supported by AutoModel.from_pretrained 
by the Transormers library (for some models, repository with the model files copied to the directory before running).

The use of other models in this benchmark is in the process of being resolved.

The list below describes the models for which it was possible to successfully obtain scores until now:

- mBERT: `bert-base-multilingual-cased`
- XLM-R: `xlm-roberta-base`
- NorBERT: `ltgoslo/norbert`
- NorBERT2: `ltgoslo/norbert2`
- NB-BERT-Base: `NbAiLab/nb-bert-base`
- Notram: `NbAiLab/notram-bert-norwegian-uncased-080321`
- Distilbert: `distilbert-base-uncased` -- there is little sense in using this model, however, an attempt was made to launch 
- ScandiBERT: `vesteinn/ScandiBERT` -- the repository with the model files has been downloaded to the directory
- XLM: `xlm-mlm-100-1280` -- RUNNING
- Bert-Base-En-Fr-De-No-Da-Cased: `Geotrend/bert-base-en-fr-de-no-da-cased` 
- La/Bse: `La/Bse`
- Electra-Small-Nordic: `jonfd/electra-small-nordic` -- IN PROGRESS (the repository with the model files has been downloaded to the directory)


---

### <a name="FINEGRAINED"></a> Fine-grained Sentiment Analysis Task

The code and overall discription for the current task can be found [in this repo](https://github.com/jerbarnes/sentiment_graphs)

---

### <a name="BINARYSENT"></a> Binary Sentiment Analysis Task

For this task, [script](https://github.com/ltgoslo/norbench/evaluation_scripts/sentiment_finetuning.py) `sentiment_finetuning.py` should be used.

#### <a name="BINARYSENT_PARAMS"></a>  Parameters

The input of the model is:

* `--model_name` - the name of the model with which the user intends to save the output file

* `--short_model_name` - the name of the model can be presented in several ways. 
  +  Firstly: in get_full_model_names, the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `mbert`
  + or the model name can be submitted as a full model name mentinoed in transformer library: `bert-base-multilingual-cased`
  
* `--use_class_weights` - a parameter that determines whether classes will be balanced when the model is running 
(classes are balanced when a `FALSE` value is passed to the parameter)

* `--training_language` - `no` or any other subdirectory in `data/sentiment/`

* `--epochs` - number of trainable epochs (`10` as default)


#### <a name="BINARYSENT_SCRIPT"></a> Running script

Scripts could be run on the [SAGA](https://documentation.sigma2.no/index.html) HPC cluster

In order to run the script on Saga, it is necessary to put arguments for [parameters](#BINARYSENT_PARAMS) in the form indicated below.

Sample `.slurm` file could be found in [this](https://github.com/ltgoslo/norbench/evaluation_scripts/evaluate_binary_sentiment.slurm) directory

```
python3 sentiment_finetuning.py --short_model_name ${MODEL} --training_language ${LANG} --model_name ${IDENTIFIER} --epochs 10 --use_class_weights $WEIGHTED
```

#### <a name="BINARYSENT_EVAL"></a>  Evaluation

F1 score is used to evaluate this task. 
The calculation of the metric takes place inside the script, so the user receives the table with the F1 scores obtained on the validation subset and on the testing data. 
The table with output scores is automatically stored in the `results` directory.

#### <a name="BINARYSENT_MODELS"></a>  Models that have been successfully tested on this script

Currently, this script can work with Bert-Like-Models, DistilBert, Xlm-Roberta models and models which are supported by AutoModel.from_pretrained 
by the Transformers library (for some models repository with the model files should be copied to the directory before running).

The use of other models in this benchmark is in the process of being resolved.

The list below describes the models for which it was possible to successfully obtain scores until now:

- mBERT: `bert-base-multilingual-cased`
- XLM-R: `xlm-roberta-base`
- NorBERT: `ltgoslo/norbert`
- NorBERT2: `ltgoslo/norbert2`
- NB-BERT-Base: `NbAiLab/nb-bert-base`
- Notram: `NbAiLab/notram-bert-norwegian-uncased-080321`
- XLM: `xlm-mlm-100-1280` -- was selected to test the possibility of launching via AutoModels
- Distilbert: `distilbert-base-uncased` -- there is little sense in using this model, however, an attempt was made to launch 
- ScandiBERT: `vesteinn/ScandiBERT` -- the repository with the model files has been downloaded to the directory
- XLM: `xlm-mlm-100-1280`
- Bert-Base-En-Fr-De-No-Da-Cased: `Geotrend/bert-base-en-fr-de-no-da-cased` 
- La/Bse: `La/Bse` -- IN PROGRESSS
- Electra-Small-Nordic: `jonfd/electra-small-nordic` -- IN PROGRESS (the repository with the model files has been downloaded to the directory)

---

### <a name="NER"></a> Named Entity Recognition Task

For this task, [script](https://github.com/ltgoslo/norbench/evaluation_scripts/ner_finetuning.py) `ner_finetuning.py` should be used.

#### <a name="NER_PARAMS"></a>  Parameters

The input of the model is: 


* `--model_name` - the name of the model can be presented in several ways.
  +  in get_full_model_names, the user can add a convenient abbreviation for the model name (if a specific model is not yet in the list): `xlm-roberta`
  + the model name can be submitted as a full model name mentioned in the Transformers library: `xlm-roberta-base`
* `--run_model_name` - the name of the model with which the user intends to save the output file
* `--training_language` - as NER task exists for both Norwegian Bokmål and Norwegian Nynorsk, `nob` or `nno` should be used respectively
* `--epochs` - number of trainable epochs (`20` by default)
* `--use_seqeval_evaluation` - boolean variable indicating whether to use the seqeval metric during validation (False by default)
* `--model_type` - optional argument (type of the model that one wants to run (`bert`, `roberta`)), 
however, if the user type does not know the model type, the decision will be made automatically, so one can skip this argument


#### <a name="NER_SCRIPT"></a> Running script

Scripts could be run on the [SAGA](https://documentation.sigma2.no/index.html) HPC cluster

In order to run the script on Saga, it is necessary to put arguments for [parameters](#NER_PARAMS) in the form indicated below.

Sample `.slurm` file could be found in [this](https://github.com/ltgoslo/norbench/evaluation_scripts/evaluate_ner.slurm) directory

```
python3 ner_finetuning.py --model_name ${MODEL}  --training_language ${LANG} --run_model_name ${IDENTIFIER} --epochs 20
```

#### <a name="NER_EVAL"></a>  Evaluation

F1 score is used to evaluate this task.
The calculation of the metric takes place inside the script, so the user receives the table with the F1 score scores obtained on the testing data. 
The table with output scores is automatically stored in the `results` directory.

NOTE: for the current task not F1 score itself is used (not the original metric). 
A special [script](https://github.com/ltgoslo/norbench/evaluation_scripts/ner_eval.py) is used to get the result on the test set 
where scores are counted for provided labels: `PER`, `ORG`, `LOC`, `GPE_LOC`, `GPE_ORG`, `PROD`, `EVT`, `DRV`. 


#### <a name="NER_MODELS"></a>  Models that have been successfully tested on this script

Currently, this script can work with Bert-Like-Models, DistilBert, Xlm-Roberta models and models which are supported by AutoModel.from_pretrained 
by the Transformers library (for some models, repository with the model files should be copied to the directory before running).

The use of other models in this benchmark is in the process of being resolved.

The list below describes the models for which it was possible to successfully obtain scores until now:

- mBERT: `bert-base-multilingual-cased`
- XLM-R: `xlm-roberta-base`
- NorBERT: `ltgoslo/norbert`
- NorBERT2: `ltgoslo/norbert2`
- NB-BERT-Base: `NbAiLab/nb-bert-base`
- Notram: `NbAiLab/notram-bert-norwegian-uncased-080321` -- IN PROGRESS 
- XLM: `xlm-mlm-100-1280` -- IN PROGRESS was selected to test the possibility of launching via AutoModels
- Distilbert: `distilbert-base-uncased` -- IN PROGRESS there is little sense in using this model, however, an attempt was made to launch 
- ScandiBERT: `vesteinn/ScandiBERT` -- the repository with the model files has been downloaded to the directory
- XLM: `xlm-mlm-100-1280`
- Bert-Base-En-Fr-De-No-Da-Cased: `Geotrend/bert-base-en-fr-de-no-da-cased` 
- La/Bse: `La/Bse`
- Electra-Small-Nordic: `jonfd/electra-small-nordic` --  the repository with the model files has been downloaded to the directory