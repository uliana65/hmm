# Hidden Markov Model (HMM)
My implementation of POS tagging with a statistical method - Hidden Markov Model. 

## Description
The method is based on three types of probabilities: 
- INITIAL $`p(tag_1|INIT)`$ : probability of a tag appearing in position 1 (sentence initial position);
- TRANSITION $`p(tag_i|{tag}_{i-1})`$ : probability of the tag in position $`i`$ appearing after the tag in position $`i-1`$; 
- EMISSION $`p(word_i|tag_i)`$ probability of word in position $`i`$ appearing together with tag in position $`i`$.

![example_sentence](https://github.com/uliana65/hmm/blob/main/figures/sent_example.png)

Given the initial, transition, and emission probabilities learnt from a corpus, the most optimal POS sequence is selected with the Viterbi algorithm. 

## Training
`python hmm.py --path PATH_TO_DATA --train_data TRAIN_DATA_FILE_NAME --test_data TEST_DATA_FILE_NAME`

Produces a test_output.tt file with model's tagging predictions. To evaluate the predictions use this command:   
`python eval.py --path PATH_TO_DATA --test_data TEST_DATA_FILE_NAME --model_pred test_output.tt`

## Results
Parsing results for a German mixed-genre dataset (see res folder):

Overall Accuracy 0.9095

By part of speech:
| POS | Precision  | Recall | F1 score | Percentage in training data |
| --- | ---------- | ------ | -------- | --------------------------- |
|NOUN|0.93|0.91|0.92|30%|
|VERB|0.92|0.92|0.92|11%|
|ADJ|0.81|0.72|0.76|7.3%|
|ADV|0.90|0.81|0.85|4.5%|
|PRON|0.87|0.84|0.85|4.9%|
|ADP|0.93|0.98|0.96|11%|
|NUM|0.99|0.77|0.86|2.7%|
|CONJ|0.95|0.90|0.92|3.7%|
|DET|0.82|0.97|0.89|11%|
|PRT|0.87|0.93|0.89|1.3%|
|PUNCT|0.96|1.00|0.98|13%|
|X|0.22|0.09|0.12|0.1%|

As you can see, it mostly struggles with the undefined category (X) which is infrequent in the training corpus (<1%). As also depicted in the graph below, more frequent categories (>1%) don't seem to pose a challenge for the model. Apart from adjectives (ADJ), F1 for every POS reaches over 80%. With ADJ, even though this POS has almost the same proportion in the test data, the model doesn't recognize this tag quite often (recall < precision). As further analysis shows, this is mainly due to the competing NOUN tag that comes more often after DET in the learnt transitional probabilities. This is not surprising as it is a typical position of adjectives in a German sentence - DET (ADJ) NOUN.

![example_sentence](https://github.com/uliana65/hmm/blob/main/figures/f1_by_distribution.png)

Overall, HMM lags behind state-of-the-art neural methods for POS tagging, but is still a helpful technique for low-resource scenarios. Looking at the graph below, it's clear that the model is able to reach a decent accuracy (~73%-85%) even on a small batch of training data (less than 2000 annotated sentences in this case). The performance went up to 90% at about 7000 examples, half of the training data set. This is even more impressive if we consider that in my implementation the data is not lemmatized hence the emission probabilities don't differentiate between forms of one and the same word and store them as separate entities.

![example_sentence](https://github.com/uliana65/hmm/blob/main/figures/accuracy_by_training_size.png)

However, among the drawbacks of HMM models for sequence tagging is its computational intensity. Although optimized, the search for the most likely tag sequence is much slower compared tagging with NN models.     
