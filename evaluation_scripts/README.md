# NorBench

##  Targeted sentiment analysis task

`python3 tsa_finetuning.py --model_name_or_path=PRETRAINED_MODEL --dataset_name="../sentiment_analysis/tsa/" --output_dir="OUT_DIR" --num_train_epochs=3 --task_name="tsa"`


---

## Sentiment classification task

### Data

1. **Sentence-level.** 

3 classes (negative, neutral and positives).

[Dataset](https://github.com/ltgoslo/norbench/tree/main/sentiment_analysis/sentence)

2. **Document-level.**

While the reviews originally come with numerical ratings on a scale of 1–6, we here conflate these to three classes; negative (1–3), fair (4), and positive (5–6). 
This mapping is done to avoid problems with too few examples for the ratings in the extreme ends of the numerical scale. 
The dataset comes with predefined datasplits (chronologically sorted). 
More information on the website of [University of Oslo](https://www.mn.uio.no/ifi/english/research/projects/sant/).

[Dataset](https://github.com/ltgoslo/norbench/tree/main/sentiment_analysis/document)


### Evaluation

`TYPE` can be either `document` or `sentence`.

For BERT-like models:

`python3 sa_classification.py  -m PRETRAINED_MODEL -i RUN_NAME --type TYPE -d TRAIN_SET -dev DEV_SET -t TEST_SET -s RANDOM_SEED`

For T5-like models:

`python3 t5_sa_classification.py  -m PRETRAINED_MODEL -i RUN_NAME --type TYPE -d TRAIN_SET -dev DEV_SET -t TEST_SET -s RANDOM_SEED`

