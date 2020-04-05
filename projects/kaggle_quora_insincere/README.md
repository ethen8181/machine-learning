# Kaggle Quora Insincere Questions

Refer to the [Kaggle competition description](https://www.kaggle.com/c/quora-insincere-questions-classification/overview) to get up to speed with the goal of this text classification competition. This competition is a kernel competition, meaning we can only use the training/test data and pre-trained embedding provided by the competition organizers.

## Documentation

- `pytorch_quora_insincere.ipynb` [[nbviewer](https://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/projects/kaggle_quora_insincere/pytorch_quora_insincere.ipynb)][[html](http://ethen8181.github.io/machine-learning/projects/kaggle_quora_insincere/pytorch_quora_insincere.html)] Four main personal learnings from this competition are:
    - Pre-trained embeddings helps with model performance if we leverage them correctly. Specifically, if we plan on using pre-trained embeddings, we should aim to get our vocabulary as close as possible to our pre-trained embedding.
    - Bucketing greatly improves the runtime. Meaning, instead of padding the entire input text data to a fixed length, we only pad the length for each batch.
    - How to use attention layer in the context of text classification.
    - Leverage pytorch framework for text classification.

## Results

Private Leaderboard Score, F1 score:

- this repo: 0.69
- leaderboard score 850th place: 0.69
- leaderboard score 1st place: 0.71

Note that the model here is not extensively tuned and no ensembling/blending/stacking was used.
  