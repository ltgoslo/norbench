# NorBench
This repository contains an emerging attempt at compiling a comprehensive set 
of NLP benchmarks for Norwegian.

We list the existing test sets, their recommended evaluation metrics 
and provide links to the evaluation code (where available). 
Currently we only link to the original datasets, 
but in the future we plan to provide their standardized versions for easier benchmarking.   

## Mainstream NLP tasks

| Task                        | Test Set               | Metrics| Evaluation code |
|-----------------------------|------------------------|--------|-----------------|
|PoS tagging                  |[Bokmaal](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal) / [Nynorsk](https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk) / [Dialects](https://github.com/UniversalDependencies/UD_Norwegian-NynorskLIA)| Macro F1 UPOS/XPOS       | [CoNLL 2018 shared task evaluation script](https://universaldependencies.org/conll18/conll18_ud_eval.py)                |
|Dependency parsing           |[Bokmaal](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal) / [Nynorsk](https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk) / [Dialects](https://github.com/UniversalDependencies/UD_Norwegian-NynorskLIA)                        |  Unlabeled/Labeled Accuracy Score (UAS/LAS)      | [CoNLL 2018 shared task evaluation script](https://universaldependencies.org/conll18/conll18_ud_eval.py)                |
|Named entity recognition     |[NorNE Bokmaal](https://github.com/ltgoslo/norne/tree/master/ud/nob) / [NorNE Nynorsk](https://github.com/ltgoslo/norne/tree/master/ud/nno)                        | Entity-level exact match F1 (strict)       | [Batista's re-implementation of the SemEval 2013 evaluation script](https://github.com/davidsbatista/NER-Evaluation)               |
|Sentence-level polarity      |[NoReC_sentence](https://github.com/ltgoslo/norec_sentence/)| Macro averaged F1      |  [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)     |
|Structured sentiment analysis|[NoReC_fine](https://github.com/ltgoslo/norec_fine)                        | Sentiment Graph F1 |  [Semeval 2022 evaluation script](https://github.com/jerbarnes/semeval22_structured_sentiment/blob/master/evaluation/evaluate_single_dataset.py)              |
|Negation cues and scopes     |[NoReC_neg](https://github.com/ltgoslo/norec_neg/)                        |  StarSem Full Negation F1      | [StarSem Perl script](https://github.com/ltgoslo/norec_neg/blob/main/modeling/evaluation/eval.cd-sco.pl)                |
|Co-reference resolution      |NARC (annotation ongoing)|MUC, $B^3$, CEAFE, LEA|[evaluation notebook NARC](https://github.com/ltgoslo/Norwegian-Coreference-Corpus/blob/f077b9de830d753b04688558c6f46157ab8fefd0/code/notebooks/evaluation.ipynb) (using [corefeval](https://github.com/tollefj/coreference-eval))|


## Lexical tasks

| Task                        | Test Set               | Metrics| Evaluation code |
|-----------------------------|------------------------|--------|-----------------|
|Synonym detection*            |[Norwegian Synonymy](https://github.com/ltgoslo/norwegian-synonyms)                        |        |                 |
|Analogical reasoning*         |[Norwegian Analogy](https://github.com/ltgoslo/norwegian-analogies)                        |        |                 |
|Word-level polarity*          |[NorSentLex](https://github.com/ltgoslo/norsentlex)                        |   Accuracy     |  [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)    |
|Word sense disambiguation in context (WSDiC)|[Norwegian WordNet](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-27/)      |Averaged macro F1 |[Preliminary example](https://github.com/ltgoslo/simple_elmo/blob/master/simple_elmo/examples/wsd_eval.py)|
|Lexical semantic change detection (LSCD)|[NorDiaChange](https://github.com/ltgoslo/nor_dia_change) | Spearman correlation, Accuracy |[SemEval'2020](https://github.com/akutuzov/semeval2020/blob/master/code/eval.py)|

## Text classification tasks

| Task                        | Test Set               | Metrics| Evaluation code |
|-----------------------------|------------------------|--------|-----------------|
|Document-level ratings         |[Norwegian Review Corpus](https://github.com/ltgoslo/norec)                        |   Macro averaged F1      |  [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)               |
|Political affiliation detection|[Talk of Norway](https://github.com/ltgoslo/talk-of-norway)                        |        |                 |
|Dialect classification in tweets  |[NorDial](https://github.com/jerbarnes/norwegian_dialect)                        |  Macro averaged F1      |  [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)               |

_* Type-based (static) models only_
