## Leaderboard for encoder-decoder models

**Important: the scores below reflect the models' performance with a specific combination of hyperparameters (see [the paper](https://aclanthology.org/2023.nodalida-1.61/) for full description).**

**We did our best to use sensible values, but your mileage may vary if you choose a different hyperparameter combination.**
**The *NorBench* leaderboard should not be considered an ultimate truth. It is rather a rough estimation of the models' capabilities.**

<table>
<tbody>
<tr class="odd">
<td style="text-align: left;"><strong>Model</strong></td>
<td style="text-align: right;"><strong>Size</strong></td>
<td style="text-align: left;"><strong>Document SA</strong></td>
<td style="text-align: left;"><strong>Sentence SA</strong></td>
<td style="text-align: left;"><strong>LingAcc</strong></td>
<td style="text-align: right;"><strong>MT</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;">
<p><a href="https://huggingface.co/ltg/nort5-xs">NorT5<sub>x-small</sub></a></p></td>
<td style="text-align: right;">32M</td>
<td style="text-align: left;"><strong>15.1</strong><span class="math inline"><sup> ± 6.4</sup></span></td>
<td style="text-align: left;"><strong>50.2</strong><span class="math inline"><sup> ± 26.5</sup></span></td>
<td style="text-align: left;"><strong>51.4</strong><span class="math inline"><sup> ± 0.4</sup></span></td>
<td style="text-align: right;"><strong>82.1</strong><span class="math inline"><sup> ± 0.2</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><a href="https://huggingface.co/ltg/nort5-small">NorT5<sub>small</sub></a></td>
<td style="text-align: right;">88M</td>
<td style="text-align: left;"><strong>36.9</strong><span class="math inline"><sup> ± 16.8</sup></span></td>
<td style="text-align: left;"><strong>51.4</strong><span class="math inline"><sup> ± 8.6</sup></span></td>
<td style="text-align: left;"><strong>54.4</strong><span class="math inline"><sup> ± 0.3</sup></span></td>
<td style="text-align: right;"><strong>85.1</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;"><a href="https://huggingface.co/google/mt5-small">mT5<sub>small</sub></a></td>
<td style="text-align: right;">300M</td>
<td style="text-align: left;">21.4<span class="math inline"><sup> ± 0.8</sup></span></td>
<td style="text-align: left;">23.4<span class="math inline"><sup> ± 1.3</sup></span></td>
<td style="text-align: left;">25.4<span class="math inline"><sup> ± 5.4</sup></span></td>
<td style="text-align: right;">33.2<span class="math inline"><sup> ± 0.3</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><a href="https://huggingface.co/north/t5_small_NCC">North-T5<sub>small</sub></a></td>
<td style="text-align: right;">300M</td>
<td style="text-align: left;">20.9<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: left;">17.5<span class="math inline"><sup> ± 0.4</sup></span></td>
<td style="text-align: left;">33.8<span class="math inline"><sup> ± 7.9</sup></span></td>
<td style="text-align: right;">36.0<span class="math inline"><sup> ± 0.1</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;"><a href="https://huggingface.co/t5-base">T5<sub>base</sub></a></td>
<td style="text-align: right;">223M</td>
<td style="text-align: left;"><strong>48.4</strong><span class="math inline"><sup> ± 3.5</sup></span></td>
<td style="text-align: left;">22.4<span class="math inline"><sup> ± 0.0</sup></span></td>
<td style="text-align: left;">17.6<span class="math inline"><sup> ± 0.8</sup></span></td>
<td style="text-align: right;">8.9<span class="math inline"><sup> ± 0.0</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><a href="https://huggingface.co/ltg/nort5-base">NorT5<sub>base</sub></a></td>
<td style="text-align: right;">228M</td>
<td style="text-align: left;"><strong>53.4</strong><span class="math inline"><sup> ± 16.7</sup></span></td>
<td style="text-align: left;"><strong>62.8</strong><span class="math inline"><sup> ± 12.6</sup></span></td>
<td style="text-align: left;"><strong>58.9</strong><span class="math inline"><sup> ± 0.3</sup></span></td>
<td style="text-align: right;"><strong>86.6</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;"><a href="https://huggingface.co/google/mt5-base">mT5<sub>base</sub></a></td>
<td style="text-align: right;">582M</td>
<td style="text-align: left;">21.0<span class="math inline"><sup> ± 0.2</sup></span></td>
<td style="text-align: left;">27.6<span class="math inline"><sup> ± 2.2</sup></span></td>
<td style="text-align: left;">25.3<span class="math inline"><sup> ± 10.1</sup></span></td>
<td style="text-align: right;">38.6<span class="math inline"><sup> ± 0.1</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><a href="https://huggingface.co/north/t5_base_NCC">North-T5<sub>base</sub></a></td>
<td style="text-align: right;">582M</td>
<td style="text-align: left;">30.5<span class="math inline"><sup> ± 21.1</sup></span></td>
<td style="text-align: left;">27.9<span class="math inline"><sup> ± 5.3</sup></span></td>
<td style="text-align: left;">41.1<span class="math inline"><sup> ± 9.6</sup></span></td>
<td style="text-align: right;">39.8<span class="math inline"><sup> ± 0.2</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;"><a href="https://huggingface.co/ltg/nort5-large">NorT5<sub>large</sub></a></td>
<td style="text-align: right;">808M</td>
<td style="text-align: left;">69.5<span class="math inline"><sup> ± 7.8</sup></span></td>
<td style="text-align: left;"><strong>63.8</strong><span class="math inline"><sup> ± 20.7</sup></span></td>
<td style="text-align: left;"><strong>59.4</strong><span class="math inline"><sup> ± 0.5</sup></span></td>
<td style="text-align: right;"><strong>86.8</strong><span class="math inline"><sup> ± 0.1</sup></span></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><a href="https://huggingface.co/google/mt5-large">mT5<sub>large</sub></a></td>
<td style="text-align: right;">1 230M</td>
<td style="text-align: left;"><strong>72.9</strong><span class="math inline"><sup> ± 2.6</sup></span></td>
<td style="text-align: left;">28.1<span class="math inline"><sup> ± 0.5</sup></span></td>
<td style="text-align: left;">50.4<span class="math inline"><sup> ± 4.0</sup></span></td>
<td style="text-align: right;">40.0<span class="math inline"><sup> ± 0.1</sup></span></td>
</tr>
<tr class="even">
<td style="text-align: left;"><a href="https://huggingface.co/north/t5_large_NCC">North-T5<sub>large</sub></a></td>
<td style="text-align: right;">1 230M</td>
<td style="text-align: left;"><strong>75.1</strong><span class="math inline"><sup> ± 5.3</sup></span></td>
<td style="text-align: left;">28.4<span class="math inline"><sup> ± 6.8</sup></span></td>
<td style="text-align: left;">46.8<span class="math inline"><sup> ± 18.7</sup></span></td>
<td style="text-align: right;">41.1<span class="math inline"><sup> ± 0.1</sup></span></td>
</tr>
</tbody>
</table>

NorBench scores for encoder-decoder models, evaluated in a generative text-to-text setting via comparing the probabilities of generated marker tokens. 
The best results (within one standard deviation) in each category are typeset in bold.

SA stands for `sentiment analysis`, LingAcc for `linguistic acceptability`, MT for `machine translation`.

All the SA models were fine-tuned with batch size 16 except `large` models which used batch size 12.

[Benchmark datasets descriptions](README.md)
