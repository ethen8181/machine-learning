- [machine-learning](#machine-learning)
  - [Documentation Listings](#documentation-listings)
    - [model deployment](#model-deployment)
    - [bandits](#bandits)
    - [search](#search)
    - [time series](#time-series)
    - [projects](#projects)
    - [ab tests](#ab-tests)
    - [model selection](#model-selection)
    - [big data](#big-data)
    - [dim reduct](#dim-reduct)
    - [recsys](#recsys)
    - [trees](#trees)
    - [clustering](#clustering)
    - [data science is software](#data-science-is-software)
    - [deep learning](#deep-learning)
    - [keras](#keras)
    - [text classification](#text-classification)
    - [networkx](#networkx)
    - [association rule](#association-rule)
    - [regularization](#regularization)
    - [ga](#ga)
    - [unbalanced](#unbalanced)
    - [clustering old](#clustering-old)
    - [linear regression](#linear-regression)
  - [Python Programming](#python-programming)

# machine-learning

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/ethen8181/machine-learning/blob/master/LICENSE)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)

This is a continuously updated repository that documents personal journey on learning data science, machine learning related topics.

**Goal:** Introduce machine learning contents in Jupyter Notebook format. The content aims to strike a good balance between mathematical notations, educational implementation from scratch using Python's scientific stack including numpy, numba, scipy, pandas, matplotlib, etc. and open-source library usage such as scikit-learn, pyspark, gensim, keras, pytorch, tensorflow, etc.


## Documentation Listings

### model deployment

- FastAPI & Azure Kubernetes Cluster. End to end example of training a model and hosting it as a service. [[folder](https://github.com/ethen8181/machine-learning/blob/master/model_deployment/fastapi_kubernetes)]

### bandits

Introduction to Multi-armed Bandits. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/bandits/multi_armed_bandits.ipynb)][[html](http://ethen8181.github.io/machine-learning/bandits/multi_armed_bandits.html)]

### search

Information Retrieval, some examples are demonstrated using ElasticSearch.

- Introduction to BM25 (Best Match). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/search/bm25_intro.ipynb)][[html](http://ethen8181.github.io/machine-learning/search/bm25_intro.html)]

### time series

Forecasting methods for timeseries-based data.

- Getting started with time series analysis with Exponential Smoothing (Holt-Winters). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/1_exponential_smoothing.ipynb)][[html](http://ethen8181.github.io/machine-learning/time_series/1_exponential_smoothing.html)]
- Framing time series problem as supervised-learning. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/3_supervised_time_series.ipynb)][[html](http://ethen8181.github.io/machine-learning/time_series/3_supervised_time_series.html)]
- First Foray Into Discrete/Fast Fourier Transformation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/fft/fft.ipynb)][[html](http://ethen8181.github.io/machine-learning/time_series/fft/fft.html)]

### projects

End to end project including data preprocessing, model building.

- [Kaggle: Don't Get Kicked](https://www.kaggle.com/c/DontGetKicked): Predict if a car purchased at auction is a unfortunate purchase. [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_dont_get_kicked/)]
- [Kaggle: Rossman Store Sales](https://www.kaggle.com/c/rossmann-store-sales/): Predicting daily store sales. Also introduces deep learning for tabular data. [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_rossman_store_sales/)]
- [Kaggle: Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification/): Predicting insincere questions. [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_quora_insincere/)]
- mlutils: Machine learning utility function package [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/mlutils/)]

### ab tests

A/B testing, a.k.a experimental design. Includes: Quick review of necessary statistic concepts. Methods and workflow/thought-process for conducting the test and caveats to look out for.

- Frequentist A/B testing (includes a quick review of concepts such as p-value, confidence interval). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/frequentist_ab_test.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html)]
- Quantile Regression and its application in A/B testing.
    - Quick Introduction to Quantile Regression. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/quantile_regression/quantile_regression.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.html)]
    - Quantile Regression's application in A/B testing. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/quantile_regression/ab_test_regression.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/ab_test_regression.html)]
- Casual Inference
    - Propensity Score Matching. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/causal_inference/matching.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/causal_inference/matching.html)]

### model selection

Methods for selecting, improving, evaluating models/algorithms.

- K-fold cross validation, grid/random search from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/model_selection.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/model_selection.html)]
- AUC (Area under the ROC curve and precision/recall curve) from scratch (includes the process of building a custom scikit-learn transformer). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/auc/auc.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/auc/auc.html)]
- Evaluation metrics for imbalanced dataset. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/imbalanced/imbalanced_metrics.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/imbalanced/imbalanced_metrics.html)]
- Detecting collinearity amongst features (Variance Inflation Factor for numeric features and Cramer's V statistics for categorical features), also introduces Linear Regression from a Maximum Likelihood perspective and the R-squared evaluation metric. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/collinearity.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/collinearity.html)]
- Curated tips and tricks for technical and soft skills. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/tips_and_tricks/tips_and_tricks.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/tips_and_tricks/tips_and_tricks.html)]
- Partial Dependence Plot (PDP), model-agnostic approach for directional feature influence. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/partial_dependence/partial_dependence.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/partial_dependence/partial_dependence.html)]
- Kullback-Leibler (KL) Divergence. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/kl_divergence.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/kl_divergence.html)]
- Probability Calibration for classification models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/prob_calibration.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html)]

### big data

Exploring big data tools, such as Spark and H2O.ai. For those interested there's also a [pyspark rdd cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_Cheat_Sheet_Python.pdf) and [pyspark dataframe cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_SQL_Cheat_Sheet_Python.pdf) that may come in handy.

- Local Hadoop cluster installation on Mac. [[markdown](https://github.com/ethen8181/machine-learning/tree/master/big_data/local_hadoop.md)]
- PySpark installation on Mac. [[markdown](https://github.com/ethen8181/machine-learning/tree/master/big_data/spark_installation.md)]
- Examples of manipulating with data (crimes data) and building a RandomForest model with PySpark MLlib. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/spark_crime.ipynb)][[html](http://ethen8181.github.io/machine-learning/big_data/spark_crime.html)]
- PCA with PySpark MLlib. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/spark_pca.ipynb)][[html](http://ethen8181.github.io/machine-learning/big_data/spark_pca.html)]
- Tuning Spark Partitions. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/spark_partitions.ipynb)][[html](http://ethen8181.github.io/machine-learning/big_data/spark_partitions.html)]
- H2O API walkthrough (using GBM as an example). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/h2o/h2o_api_walkthrough.ipynb)][[html](http://ethen8181.github.io/machine-learning/big_data/h2o/h2o_api_walkthrough.html)]
- Spark MLlib Binary Classification (using GBM as an example). [[raw zeppelin notebook](https://github.com/ethen8181/machine-learning/blob/master/big_data/sparkml/sparkml.json)][[Zepl](https://www.zepl.com/explore)]

### dim reduct

Dimensionality reduction methods.

- Principal Component Analysis (PCA) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/dim_reduct/PCA.ipynb)][[html](http://ethen8181.github.io/machine-learning/dim_reduct/PCA.html)]
- Introduction to Singular Value Decomposition (SVD), also known as Latent Semantic Analysis/Indexing (LSA/LSI). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/dim_reduct/svd.ipynb)][[html](http://ethen8181.github.io/machine-learning/dim_reduct/svd.html)]

### recsys

Recommendation system with a focus on matrix factorization methods. Starters into the field should go through the first notebook to understand the basics of matrix factorization methods.

- Alternating Least Squares with Weighted Regularization (ALS-WR) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/1_ALSWR.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/1_ALSWR.html)]
- ALS-WR for implicit feedback data from scratch & Mean Average Precision at k (mapk) and Normalized Cumulative Discounted Gain (ndcg) evaluation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/2_implicit.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/2_implicit.html)]
- Bayesian Personalized Ranking (BPR) from scratch & AUC evaluation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/4_bpr.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/4_bpr.html)]
- WARP (Weighted Approximate-Rank Pairwise) Loss using lightfm. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/5_warp.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/5_warp.html)]
- Factorization Machine from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/factorization_machine/factorization_machine.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/factorization_machine/factorization_machine.html)]
- Content-Based Recommenders:
    - (Text) Content-Based Recommenders. Introducing Approximate Nearest Neighborhood (ANN) - Locality Sensitive Hashing (LSH) for cosine distance from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/content_based/lsh_text.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/content_based/lsh_text.html)]
- Approximate Nearest Neighborhood (ANN):
    -  Benchmarking ANN implementations (nmslib). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/ann_benchmarks/ann_benchmarks.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/ann_benchmarks/ann_benchmarks.html)]
- Calibrated Recommendation for reducing bias/increasing diversity in recommendation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/calibration/calibrated_reco.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/calibration/calibrated_reco.html)]
- Maximum Inner Product for Speeding Up Generating Recommendations. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/max_inner_product/max_inner_product.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/max_inner_product/max_inner_product.html)]

### trees

Tree-based models for both regression and classification tasks.

- Decision Tree from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/decision_tree.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/decision_tree.html)]
- Random Forest from scratch and Extra Trees. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/random_forest.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/random_forest.html)]
- Gradient Boosting Machine (GBM) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/gbm/gbm.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/gbm/gbm.html)]
- Xgboost API walkthrough (includes hyperparameter tuning via scikit-learn like API). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/xgboost.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/xgboost.html)]
- LightGBM API walkthrough and a discussion about categorical features in tree-based models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/lightgbm.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/lightgbm.html)]
- Monotonic Constraint with Boosted Tree. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/monotonic.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/monotonic.html)]

### clustering

TF-IDF and Topic Modeling are techniques specifically used for text analytics.

- TF-IDF (text frequency - inverse document frequency) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/tfidf/tfidf.ipynb)][[html](http://ethen8181.github.io/machine-learning/clustering/tfidf/tfidf.html)]
- K-means, K-means++ from scratch; Elbow method for choosing K. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/kmeans.ipynb)][[html](http://ethen8181.github.io/machine-learning/clustering/kmeans.html)]
- Gaussian Mixture Model from scratch; AIC and BIC for choosing the number of Gaussians. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/GMM/GMM.ipynb)][[html](http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html)]
- Topic Modeling with gensim's Latent Dirichlet Allocation(LDA). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/topic_model/LDA.ipynb)][[html](http://ethen8181.github.io/machine-learning/clustering/topic_model/LDA.html)]

### data science is software  

Best practices for doing data science in Python.

- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/data_science_is_software/notebooks/data_science_is_software.ipynb)][[html](http://ethen8181.github.io/machine-learning/data_science_is_software/notebooks/data_science_is_software.html)]

### deep learning

Curated notes on deep learning.

- Softmax Regression from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/softmax.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/softmax.html)]
- Softmax Regression - Tensorflow hello world. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/softmax_tensorflow.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/softmax_tensorflow.html)]
- Multi-layers Neural Network - Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/nn_tensorflow.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/nn_tensorflow.html)]
- Convolutional Neural Network (CNN) - Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/cnn_image_tensorflow.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/cnn_image_tensorflow.html)]
- Recurrent Neural Network (RNN).
    - Vanilla RNN - Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/rnn/1_tensorflow_rnn.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/rnn/1_tensorflow_rnn.html)]
    - Long Short Term Memory (LSTM) - Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/rnn/2_tensorflow_lstm.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/rnn/2_tensorflow_lstm.html)]
    - RNN, LSTM - PyTorch hello world. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/rnn/1_pytorch_rnn.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/rnn/1_pytorch_rnn.html)]
- Word2vec (skipgram + negative sampling) using Gensim. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/word2vec/word2vec_detailed.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/word2vec/word2vec_detailed.html)]
- Sequence to Sequence Neural Network (Seq2Seq).
    - Seq2Seq for German to English Machine Translation - PyTorch. Includes quick intro to torchtext [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/1_torch_seq2seq_intro.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/1_torch_seq2seq_intro.html)]
    - Seq2Seq with Attention for German to English Machine Translation - PyTorch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/2_torch_seq2seq_attention.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/2_torch_seq2seq_attention.html)]
- Subword Tokenization.
    - Byte Pair Encoding (BPE) from scratch and quick walkthrough of sentencepiece. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/subword/bpe.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html)]
- Fasttext.
    - MultiLabel Text Classification with Fasttext and Huggingface Tokenizers. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/fasttext.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/fasttext.html)]
    - Product Quantization for Model Compression. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/product_quantization.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/product_quantization.html)]
    - Approximate Nearest Neighborhood Search with Navigable Small World. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/nsw.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/nsw.html)]

### keras

For those interested there's also a [keras cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Keras_Cheat_Sheet_Python.pdf) that may come in handy.

- Multi-layers Neural Network (keras basics). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/nn_keras_basics.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/nn_keras_basics.html)]
- Multi-layers Neural Network hyperparameter tuning via scikit-learn like API. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/nn_keras_hyperparameter_tuning.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/nn_keras_hyperparameter_tuning.html)]
- Convolutional Neural Network (CNN)
    - Image classification basics. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/cnn_image_keras.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/cnn_image_keras.html)]
    - Introduction to Residual Networks (ResNets) and Class Activation Maps (CAM). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/resnet_cam/resnet_cam.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/resnet_cam/resnet_cam.html)]
- Recurrent Neural Network (RNN) - language modeling basics. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/rnn_language_model_basic_keras.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/rnn_language_model_basic_keras.html)]
- Text Classification
    - Word2vec for Text Classification. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/word2vec_text_classification.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html)]
    - Leveraging Pre-trained Word Embedding for Text Classification. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/keras_pretrained_embedding.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/text_classification/keras_pretrained_embedding.html)]
    - Sentencepiece Subword tokenization for Text Classification. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/keras_subword_tokenization.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/text_classification/keras_subword_tokenization.html)]

### text classification

Deep learning techniques for text classification are categorized in its own section.

- Building intuition with spam classification using scikit-learn (scikit-learn hello world). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/basics/basics.ipynb)][[html](http://ethen8181.github.io/machine-learning/text_classification/basics/basics.html)]
- Bernoulli and Multinomial Naive Bayes from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/naive_bayes/naive_bayes.ipynb)][[html](http://ethen8181.github.io/machine-learning/text_classification/naive_bayes/naive_bayes.html)]
- Logistic Regression (stochastic gradient descent) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/logistic.ipynb)][[html](http://ethen8181.github.io/machine-learning/text_classification/logistic.html)]
- Chi-square feature selection from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/chisquare.ipynb)][[html](http://ethen8181.github.io/machine-learning/text_classification/chisquare.html)]

### networkx

Graph library other than `networkx` are also discussed.

- PyCon 2016: Practical Network Analysis Made Simple. Quickstart to networkx's API. Includes some basic graph plotting and algorithms. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/networkx/networkx.ipynb)][[html](http://ethen8181.github.io/machine-learning/networkx/networkx.html)]
- Short Walkthrough of PageRank. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/networkx/page_rank.ipynb)][[html](http://ethen8181.github.io/machine-learning/networkx/page_rank.html)]
- Influence Maximization from scratch. Includes discussion on Independent Cascade (IC), Submodular Optimization algorithms including Greedy and Lazy Greedy, a.k.a Cost Efficient Lazy Forward (CELF) [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/networkx/max_influence/max_influence.ipynb)][[html](http://ethen8181.github.io/machine-learning/networkx/max_influence/max_influence.html)]

### association rule

Also known as market-basket analysis.

- Apriori from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/association_rule/apriori.ipynb)][[html](http://ethen8181.github.io/machine-learning/association_rule/apriori.html)]
- Using R's arules package (aprori) on tabular data. [[Rmarkdown](http://ethen8181.github.io/machine-learning/association_rule/R/apriori.html)]

### regularization

Building intuition on Ridge and Lasso regularization using scikit-learn.
 
- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/regularization/regularization.ipynb)][[html](http://ethen8181.github.io/machine-learning/regularization/regularization.html)]

### ga

Genetic Algorithm. Math-free explanation and code from scratch.

- Start from a simple optimization problem and extending it to traveling salesman problem (tsp).
- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ga/ga.ipynb)][[html](http://ethen8181.github.io/machine-learning/ga/ga.html)]

### unbalanced

Choosing the optimal cutoff value for logistic regression using cost-sensitive mistakes (meaning when the cost of misclassification might differ between the two classes) when your dataset consists of unbalanced binary classes. e.g. Majority of the data points in the dataset have a positive outcome, while few have negative, or vice versa. The notion can be extended to any other classification algorithm that can predict class’s probability, this documentation just uses logistic regression for illustration purpose.

- Visualize two by two standard confusion matrix and ROC curve with costs using ggplot2.
- View [[Rmarkdown](http://ethen8181.github.io/machine-learning/unbalanced/unbalanced.html)]

### clustering old

A collection of scattered old clustering documents in R.

- Toy sample code of the LDA algorithm (gibbs sampling) and the topicmodels library. [[Rmarkdown](http://ethen8181.github.io/machine-learning/clustering_old/topic_model/LDA.html)]
- k-shingle, Minhash and Locality Sensitive Hashing for solving the problem of finding textually similar documents. [[Rmarkdown](http://ethen8181.github.io/machine-learning/clustering_old/text_similarity/text_similarity.html)]
- Introducing tf-idf (term frequency-inverse document frequency), a text mining technique. Also uses it to perform text clustering via hierarchical clustering. [[Rmarkdown](http://ethen8181.github.io/machine-learning/clustering_old/tf_idf/tf_idf.html)]
- Some useful evaluations when working with hierarchical clustering and K-means clustering (K-means++ is used here). Including Calinski-Harabasz index for determine the right K (cluster number) for clustering and boostrap evaluation of the clustering result’s stability. [[Rmarkdown](http://ethen8181.github.io/machine-learning/clustering_old/clustering/clustering.html)]

### linear regression

- Training Linear Regression with gradient descent in R, briefly covers the interpretation and visualization of linear regression's summary output. [[Rmarkdown](http://ethen8181.github.io/machine-learning/linear_regression/linear_regession.html)]


## Python Programming

- Extremely Quick Guide to Unicode. [[markdown](https://github.com/ethen8181/machine-learning/blob/master/python/unicode.md)]
- Quick Example of Factory Design Pattern. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/factory_pattern.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/factory_pattern.html)]
- Parallel programming with Python (threading, multiprocessing, concurrent.futures, joblib). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/parallel.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/parallel.html)]
- Understanding iterables, iterator and generators. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/iterator/iterator.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/iterator/iterator.html)]
- Cohort analysis. Visualizing user retention by cohort with seaborn's heatmap and illustrating pandas's unstack. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/cohort/cohort.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/cohort/cohort.html)]
- Logging module. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/logging.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/logging.html)]
- Data structure, algorithms from scratch. [[folder](https://github.com/ethen8181/machine-learning/tree/master/python/algorithms)]
- Cython and Numba quickstart for high performance Python. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/cython/cython.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/cython/cython.html)]
- Optimizing Pandas (e.g. reduce memory usage using category type). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pandas/pandas.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/pandas/pandas.html)]
- Unittest. [[Python script](https://github.com/ethen8181/machine-learning/blob/master/python/test.py)]
- Using built-in data structure and algorithm. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/1_data_structure.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/python3_cookbook/1_data_structure.html)]
- Tricks with strings and text. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/2_strings_and_text.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/python3_cookbook/2_strings_and_text.html)]
- Python's decorators (useful script for logging and timing function). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/decorators/decorators.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/decorators/decorators.html)]
- Pandas's pivot table. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pivot_table/pivot_table.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/pivot_table/pivot_table.html)]
- Quick introduction to classmethod, staticmethod and property. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/class.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/class.html)]
