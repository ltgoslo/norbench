
# Document and sentence-level sentiment analysis for Norwegian text.

Information about project you an fine on the website of [University of Oslo](https://www.mn.uio.no/ifi/english/research/projects/sant/)

For sentence level there are 3 classes:
- negative
- neutral
- positive

For document level we also 3 distinguish classes but with different meaning:
- negative
- fair 
- positive

# How to fine-tune?

<br>

For this run finetune.py and specify required arguments:
- **-level**: 'sentence' if you want to use corpora with sentence-level sentiment analysys or 'document' for document-level SA. 
- **-model**: pre-traied model from [huggingface](https://huggingface.co/models) or absolute (!) path to local folder with config (.json) and model (.bin) in case you want to use custom wrapper.

<br>
If you want to use custom wrapper instead of huggingface, please specify:

- **-custom_wrapper** = True. It's False by default

<br>

If you want to use T5 model, please specify:

- **-t5** = True. It's False by default.

<br>


There are also additional arguments possible but not required:

- **-data_path**: path to folder with train.csv, dev.csv and test.csv datasets. If no path is specified datasets will be downloaded from [repo for sentence-level SA](https://github.com/ltgoslo/norec_sentence) or [repo for document-level SA](https://github.com/ltgoslo/norec)  depending on the level you specified in the first argument. Repos will only be downloaded once, created dataframes will be stored in 'data/documnent/' and 'data/sentence' and will be used in the future experiments (no need to specify this path, script finds it automatically).
- **-lr**: 1e-05 by default.
- **-max_length**: 512 by default.
- **-warmup**: 2 by default.
- **-batch_size**: 4 by default.
- **-epochs**: 10 by default


