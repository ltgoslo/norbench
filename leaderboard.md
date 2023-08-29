## Leaderboard for encoder models

**Important: the scores below reflect the models' performance with a specific combination of hyperparameters (see [the paper](https://aclanthology.org/2023.nodalida-1.61/) for full description).**

**We did our best to use sensible values, but your mileage may vary if you choose a different hyperparameter combination.**
**The *NorBench* leaderboard should not be considered an ultimate truth. It is rather a rough estimation of the models' capabilities.**

<table>
<tbody>
<tr class="odd">
<td style="text-align: left;">
<p><strong>Model</strong></p></td>
<td style="text-align: right;"><strong>Size</strong></td>
<td style="text-align: right;"><strong>POS</strong></td>
<td style="text-align: right;"><strong>Feats</strong></td>
<td style="text-align: right;"><strong>Lemma</strong></td>
<td style="text-align: right;"><strong>DP</strong></td>
<td style="text-align: right;"><strong>NER</strong></td>
<td style="text-align: right;"><strong>Doc. SA</strong></td>
<td style="text-align: right;"><strong>Sent. SA</strong></td>
<td style="text-align: right;"><strong>TSA</strong></td>
<td style="text-align: right;"><strong>LingAcc</strong></td>
<td style="text-align: right;"><strong>QA</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;">
<p><a href="https://huggingface.co/ltg/norbert3-xs">NorBERT<sub>3, x-small</sub></a></p></td>
<td style="text-align: right;">15M</td>
<td style="text-align: right;"><strong>98.8</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>97.0</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>97.6</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>92.2</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>86.3</strong><span class="math inline"><sup> ± 0.4</sup></span></td>
<td style="text-align: right;"><strong>69.6</strong><span class="math inline"><sup> ± 2.4</sup></span></td>
<td style="text-align: right;"><strong>66.2</strong><span class="math inline"><sup> ± 1.2</sup></span></td>
<td style="text-align: right;"><strong>43.2</strong><span class="math inline"><sup> ± 0.5</sup></span></td>
<td style="text-align: right;"><strong>47.1</strong><span class="math inline"><sup> ± 0.5</sup></span></td>
<td style="text-align: right;"><strong>65.6</strong><span class="math inline"><sup> ± 3.9</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;">
<p><a href="https://huggingface.co/ltg/norbert3-small">NorBERT<sub>3, small</sub></a></p></td>
<td style="text-align: right;">40M</td>
<td style="text-align: right;"><strong>98.9</strong><span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>97.9</strong><span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>98.3</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>93.7</strong><span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>89.0</strong><span class="math inline"><sup> ± 0.3</sup></span></td>
<td style="text-align: right;"><strong>74.4</strong><span class="math inline"><sup> ± 0.5</sup></span></td>
<td style="text-align: right;"><strong>71.9</strong><span class="math inline"><sup> ± 1.3</sup></span></td>
<td style="text-align: right;"><strong>48.9</strong><span class="math inline"><sup> ± 0.9</sup></span></td>
<td style="text-align: right;"><strong>55.9</strong><span class="math inline"><sup> ± 0.2</sup></span></td>
<td style="text-align: right;"><strong>80.5</strong><span class="math inline"><sup> ± 1.2</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;">
<p><a href="https://huggingface.co/bert-base-cased">BERT<sub>base, cased</sub></a></p></td>
<td style="text-align: right;">111M</td>
<td style="text-align: right;">97.9<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">96.4<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">97.9<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">89.8<span class="math inline"><sup> ± 0.2</sup></span></td>
<td style="text-align: right;">73.4<span class="math inline"><sup> ± 0.7</sup></span></td>
<td style="text-align: right;">57.3<span class="math inline"><sup> ± 1.4</sup></span></td>
<td style="text-align: right;">53.0<span class="math inline"><sup> ± 1.1</sup></span></td>
<td style="text-align: right;">23.2<span class="math inline"><sup> ± 2.2</sup></span></td>
<td style="text-align: right;">23.9<span class="math inline"><sup> ± 0.4</sup></span></td>
<td style="text-align: right;">44.9<span class="math inline"><sup> ± 2.2</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;">
<p><a href="https://huggingface.co/ltg/norbert">NorBERT<sub>1</sub></a></p></td>
<td style="text-align: right;">111M</td>
<td style="text-align: right;">98.8<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">97.8<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">98.5<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">93.3<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">86.9<span class="math inline"><sup> ± 0.9</sup></span></td>
<td style="text-align: right;">70.1<span class="math inline"><sup> ± 0.4</sup></span></td>
<td style="text-align: right;">70.7<span class="math inline"><sup> ± 0.9</sup></span></td>
<td style="text-align: right;">45.4<span class="math inline"><sup> ± 1.1</sup></span></td>
<td style="text-align: right;">35.9<span class="math inline"><sup> ± 1.7</sup></span></td>
<td style="text-align: right;">72.5<span class="math inline"><sup> ± 1.6</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;">
<p><a href="https://huggingface.co/ltg/norbert3-base">NorBERT<sub>3, base</sub></a></p></td>
<td style="text-align: right;">123M</td>
<td style="text-align: right;"><strong>99.0</strong><span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>98.3</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">98.8<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>94.2</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>89.4</strong><span class="math inline"><sup> ± 0.9</sup></span></td>
<td style="text-align: right;"><strong>76.2</strong><span class="math inline"><sup> ± 0.8</sup></span></td>
<td style="text-align: right;"><strong>74.4</strong><span class="math inline"><sup> ± 0.3</sup></span></td>
<td style="text-align: right;"><strong>50.2</strong><span class="math inline"><sup> ± 0.7</sup></span></td>
<td style="text-align: right;"><strong>59.2</strong><span class="math inline"><sup> ± 0.3</sup></span></td>
<td style="text-align: right;"><strong>86.2</strong><span class="math inline"><sup> ± 0.3</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;">
<p><a href="https://huggingface.co/ltg/norbert2">NorBERT<sub>2</sub></a></p></td>
<td style="text-align: right;">125M</td>
<td style="text-align: right;">98.7<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">97.6<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">98.2<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">93.4<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">85.0<span class="math inline"><sup> ± 0.9</sup></span></td>
<td style="text-align: right;">73.5<span class="math inline"><sup> ± 1.1</sup></span></td>
<td style="text-align: right;">72.5<span class="math inline"><sup> ± 1.5</sup></span></td>
<td style="text-align: right;">45.4<span class="math inline"><sup> ± 1.1</sup></span></td>
<td style="text-align: right;">56.1<span class="math inline"><sup> ± 0.3</sup></span></td>
<td style="text-align: right;">76.6<span class="math inline"><sup> ± 0.7</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;">
<p><a href="https://huggingface.co/vesteinn/ScandiBERT">ScandiBERT</a></p></td>
<td style="text-align: right;">124M</td>
<td style="text-align: right;">98.9<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">98.1<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">98.7<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>94.1</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>89.4</strong><span class="math inline"><sup> ± 0.5</sup></span></td>
<td style="text-align: right;">73.9<span class="math inline"><sup> ± 0.4</sup></span></td>
<td style="text-align: right;">71.6<span class="math inline"><sup> ± 1.3</sup></span></td>
<td style="text-align: right;">48.8<span class="math inline"><sup> ± 1.0</sup></span></td>
<td style="text-align: right;">57.1<span class="math inline"><sup> ± 0.4</sup></span></td>
<td style="text-align: right;">79.0<span class="math inline"><sup> ± 0.7</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;">
<p><a href="https://huggingface.co/NbAiLab/nb-bert-base">NB-BERT<sub>base</sub></a></p></td>
<td style="text-align: right;">178M</td>
<td style="text-align: right;">98.9<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>98.3</strong><span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>98.9</strong><span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>94.1</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>89.6</strong><span class="math inline"><sup> ± 0.9</sup></span></td>
<td style="text-align: right;">74.3<span class="math inline"><sup> ± 0.6</sup></span></td>
<td style="text-align: right;">73.7<span class="math inline"><sup> ± 0.8</sup></span></td>
<td style="text-align: right;">49.2<span class="math inline"><sup> ± 1.3</sup></span></td>
<td style="text-align: right;">58.1<span class="math inline"><sup> ± 0.5</sup></span></td>
<td style="text-align: right;">79.1<span class="math inline"><sup> ± 1.2</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;">
<p><a href="https://huggingface.co/bert-base-multilingual-cased">mBERT</a></p></td>
<td style="text-align: right;">178M</td>
<td style="text-align: right;">98.4<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">97.3<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">98.3<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">92.2<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">83.5<span class="math inline"><sup> ± 0.6</sup></span></td>
<td style="text-align: right;">67.9<span class="math inline"><sup> ± 1.2</sup></span></td>
<td style="text-align: right;">62.7<span class="math inline"><sup> ± 1.2</sup></span></td>
<td style="text-align: right;">39.6<span class="math inline"><sup> ± 1.3</sup></span></td>
<td style="text-align: right;">46.4<span class="math inline"><sup> ± 0.7</sup></span></td>
<td style="text-align: right;">76.5<span class="math inline"><sup> ± 0.9</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;">
<p><a href="https://huggingface.co/xlm-roberta-base">XLM-R<sub>base</sub></a></p></td>
<td style="text-align: right;">278M</td>
<td style="text-align: right;">98.8<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">97.7<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">98.7<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">93.7<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">87.6<span class="math inline"><sup> ± 0.6</sup></span></td>
<td style="text-align: right;">73.1<span class="math inline"><sup> ± 0.7</sup></span></td>
<td style="text-align: right;">72.2<span class="math inline"><sup> ± 0.3</sup></span></td>
<td style="text-align: right;">49.4<span class="math inline"><sup> ± 0.5</sup></span></td>
<td style="text-align: right;">58.6<span class="math inline"><sup> ± 0.3</sup></span></td>
<td style="text-align: right;">78.9<span class="math inline"><sup> ± 0.6</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;">
<p><a href="https://huggingface.co/ltg/norbert3-large">NorBERT<sub>3, large</sub></a></p></td>
<td style="text-align: right;">353M</td>
<td style="text-align: right;"><strong>99.1</strong><span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>98.5</strong><span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>99.1</strong><span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;"><strong>94.6</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>91.4</strong><span class="math inline"><sup> ± 0.5</sup></span></td>
<td style="text-align: right;"><strong>79.2</strong><span class="math inline"><sup> ± 0.7</sup></span></td>
<td style="text-align: right;"><strong>78.4</strong><span class="math inline"><sup> ± 0.6</sup></span></td>
<td style="text-align: right;"><strong>54.1</strong><span class="math inline"><sup> ± 0.6</sup></span></td>
<td style="text-align: right;"><strong>61.0</strong><span class="math inline"><sup> ± 0.4</sup></span></td>
<td style="text-align: right;"><strong>88.7</strong><span class="math inline"><sup> ± 0.8</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;">
<p><a href="https://huggingface.co/NbAiLab/nb-bert-large">NB-BERT<sub>large</sub></a></p></td>
<td style="text-align: right;">355M</td>
<td style="text-align: right;">98.7<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">98.2<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">98.3<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;"><strong>94.6</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">89.8<span class="math inline"><sup> ± 0.6</sup></span></td>
<td style="text-align: right;"><strong>79.2</strong><span class="math inline"><sup> ± 0.9</sup></span></td>
<td style="text-align: right;">77.5<span class="math inline"><sup> ± 0.7</sup></span></td>
<td style="text-align: right;"><strong>54.6</strong><span class="math inline"><sup> ± 0.7</sup></span></td>
<td style="text-align: right;">59.7<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">87.0<span class="math inline"><sup> ± 0.5</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;">
<p><a href="https://huggingface.co/xlm-roberta-large">XLM-R<sub>large</sub></a></p></td>
<td style="text-align: right;">560M</td>
<td style="text-align: right;">98.9<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">98.0<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: right;">98.8<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">94.3<span class="math inline"><sup> ± 0.1</sup></span></td>
<td style="text-align: right;">87.5<span class="math inline"><sup> ± 1.0</sup></span></td>
<td style="text-align: right;">76.8<span class="math inline"><sup> ± 0.6</sup></span></td>
<td style="text-align: right;">75.4<span class="math inline"><sup> ± 1.3</sup></span></td>
<td style="text-align: right;">52.3<span class="math inline"><sup> ± 0.6</sup></span></td>
<td style="text-align: right;">58.6<span class="math inline"><sup> ± 0.3</sup></span></td>
<td style="text-align: right;">84.8<span class="math inline"><sup> ± 0.5</sup></span></td>
</tr>

</tbody>
</table>

NorBench scores for different language models. We report the mean and standard deviation
statistics over 5 runs. The
'Size' column reports the number of parameters in the model; the models
are sorted by this value. The best
results (within one standard deviation) in each size category are typeset in
bold.

SA stands for `sentiment analysis`, DP for `dependency parsing`, LingAcc for `linguistic acceptability`, QA for `question answering`.

[Benchmark datasets descriptions](README.md)