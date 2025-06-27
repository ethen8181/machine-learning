# Changelog

The changelog will record what content was **changed** (e.g. changed an existing paragraph to a better-worded version, re-ran the notebook using an updated version of the package, introduced new content to existing notebook), **added** (e.g. a completely new jupyter notebook).

## [2025-06]

### Added

- LLM Batch Inference with Ray and VLLM (2D parallelism, data + tensor). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/llm_batch_inference/llm_batch_inference_ray_vllm.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/llm_batch_inference/llm_batch_inference_ray_vllm.html)]

## [2024-10]

### Added

- Direct Preference Optimization (DPO) [[nbviewer](https://nbviewer.org/github/ethen8181/machine-learning/blob/master/deep_learning/llm/rlhf/dpo.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/llm/rlhf/dpo.html)]
- LLM Pairwise Judge (PyTorch Lightning) [[nbviewer](https://nbviewer.org/github/ethen8181/machine-learning/blob/master/deep_learning/llm/judge/llm_pairwise_judge.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/llm/judge/llm_pairwise_judge.html)]

## [2024-03]

### Added

- Multilingual Sentence Embedding with LLM and PEFT LoRA (PyTorch Lightning) [[nbviewer](https://nbviewer.org/github/ethen8181/machine-learning/blob/master/deep_learning/contrastive/sentence_embedding_peft/sentence_embedding_peft.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/contrastive/sentence_embedding_peft/sentence_embedding_peft.html)]

## [2023-11]

### Changed

- Introduction to CLIP (Contrastive Language-Image Pre-training), LiT, ViT [[nbviewer](https://nbviewer.org/github/ethen8181/machine-learning/blob/master/deep_learning/contrastive/clip/clip.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/contrastive/clip/clip.html)]
  - Massive overhaul to the content using latest version of PyTorch 2.
  - Switched to using huggingface transformer ViT image encoder, instead of timm's ResNet.
  - Added quantitative evaluation with retrieval recall@k.
  - Added additional introduction to LiT, ViT.


## [2023-10]

### Added

- BERT CTR. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/tabular/bert_ctr/bert_ctr.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/tabular/bert_ctr/bert_ctr.html)]

## [2023-09]

### Added

- Deep Learning - Learning to Rank 101 (RankNet, ListNet). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/tabular/deep_learning_learning_to_rank.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/tabular/deep_learning_learning_to_rank.html)]

## [2023-08]

### Changed

- Finetuning Pre-trained BERT Model on Text Classification Task And Inferencing with ONNX Runtime. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/transformers/text_classification_onnxruntime.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/transformers/text_classification_onnxruntime.html)]
  - Overhaul the content using latest version of PyTorch 2.
  - Use huggingface trainer for model training/evaluation instead of custom implementation.
  - Benchmarked ONNX versus PyTorch on both GPU and CPU.
- Deep Learning for Tabular Data - PyTorch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/tabular/deep_learning_tabular.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/tabular/deep_learning_tabular.html)]
  - Overhaul the content using latest version of PyTorch 2.
  - Use huggingface trainer for model training/evaluation instead of custom implementation.
  - Removed outdated content around ONNX which are not relevant for this particular topic.

## [2023-07]

### Added

- Self Supervised (SIMCLR) versus Supervised Contrastive Learning. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/contrastive/self_supervised_vs_supervised_contrastive/self_supervised_vs_supervised_contrastive.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/contrastive/self_supervised_vs_supervised_contrastive/self_supervised_vs_supervised_contrastive.html)]

## [2023-04]

### Changed

- Training Bi-Encoder Models with Contrastive Learning Notes. [[nbviewer](https://nbviewer.org/github/ethen8181/machine-learning/blob/master/deep_learning/contrastive/contrastive_learning_notes.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/contrastive/contrastive_learning_notes.html)]
  - Update to include a section on data augmentation as well as various clarification on wordings.
- Response Knowledge Distillation for Training Student Model. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/transformers/response_knowledge_distillation.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/transformers/response_knowledge_distillation.html)]
  - Re-ran with huggingface dataset logging disabled, this is to prevent messages from flooding main content.
- Sentence Transformer: Training Bi-Encoder via Contrastive Loss. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/contrastive/sentence_transformer.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/contrastive/sentence_transformer.html)]
  - Re-ran with huggingface dataset logging disabled, this is to prevent messages from flooding main content.

## [2023-03]

### Added

- Uploading and downloading files from s3. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/aws/aws_s3.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/aws/aws_s3.html)]

## [2023-02]

### Added

- Training Bi-Encoder Models with Contrastive Learning Notes. [[nbviewer](https://nbviewer.org/github/ethen8181/machine-learning/blob/master/deep_learning/contrastive/contrastive_learning_notes.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/contrastive/contrastive_learning_notes.html)]
- Introduction to CLIP (Contrastive Language-Image Pre-training) [[nbviewer](https://nbviewer.org/github/ethen8181/machine-learning/blob/master/deep_learning/contrastive/clip/clip.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/contrastive/clip/clip.html)]

## [2023-01]

### Changed

- Machine Translation with Huggingface Transformers mT5. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/translation_mt5/translation_mt5.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/translation_mt5/translation_mt5.html)]
  - This is a complete overhaul of the original Machine Translation with Huggingface Transformers article, which was a bit obsolete.
- Response Knowledge Distillation for Training Student Model. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/transformers/response_knowledge_distillation.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/transformers/response_knowledge_distillation.html)]
  - Added a final notes section on distilbert, well read students read well.

## [2022-12]

### Added

- Fine Tuning Pre-trained Encoder on Question Answer Task. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/question_answer/question_answer.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/question_answer/question_answer.html)]

## [2022-11]

### Added

- Sentence Transformer: Training Bi-Encoder via Contrastive Loss. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/contrastive/sentence_transformer.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/contrastive/sentence_transformer.html)]

## [2022-10]

### Added

- Quick Introduction to Graph Neural Network Node Classification Task (DGL, GraphSAGE). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/gnn/gnn_node_classification_intro.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/gnn/gnn_node_classification_intro.html)]

## [2022-09]

### Added

- Response Knowledge Distillation for Training Student Model. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/transformers/response_knowledge_distillation.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/transformers/response_knowledge_distillation.html)]

## [2022-07]

### Added

- HyperParameter Tuning with Ray Tune and Hyperband. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/ray_tune_hyperband.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/ray_tune_hyperband.html)]

## [2022-06]

### Added

- Quick introduction to difference in difference. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/causal_inference/diff_in_diff.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/causal_inference/diff_in_diff.html)]

### Changed

- Quick Intro to Gradient Boosted Tree Inferencing. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/gbt_inference/gbt_inference.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/gbt_inference/gbt_inference.html)]
    - Added content around ONNX.

## [2022-04]

### Added

- Quick introduction to generalized second price auction. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ad/gsp_ad_auction.ipynb)][[html](http://ethen8181.github.io/machine-learning/ad/gsp_ad_auction.html)]

## [2021-10]

### Added

- Operation Research Quick Intro Via Ortools. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/operation_research/ortools.ipynb)][[html](http://ethen8181.github.io/machine-learning/operation_research/ortools.html)]

## [2021-09]

### Added

- Probability Calibration for deep learning classification models with Temperature Scaling. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/deeplearning_prob_calibration.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/prob_calibration/deeplearning_prob_calibration.html)]

## [2021-06]

### Added

- Finetuning Pre-trained BERT Model on Text Classification Task And Inferencing with ONNX Runtime. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/transformers/text_classification_onnxruntime.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/transformers/text_classification_onnxruntime.html)]

## [2021-05]

### Added

- Machine Translation with Huggingface Transformers. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/huggingface_torch_transformer.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/huggingface_torch_transformer.html)]

## [2021-02]

### Added

- Quick Intro to Gradient Boosted Tree Inferencing. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/gbt_inference/gbt_inference.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/gbt_inference/gbt_inference.html)]

## [2021-01]

### Added

- Transformer, Attention is All you Need - PyTorch, Huggingface Datasets. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/torch_transformer.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/torch_transformer.html)]

## [2020-11]

### Added

- Inverse Propensity Weighting. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/causal_inference/inverse_propensity_weighting.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/causal_inference/inverse_propensity_weighting.html)]

## [2020-10]

### Added

- Deep Learning for Tabular Data - PyTorch, PyTorch Lightning, ONNX Runtime. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/tabular/tabular.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/tabular/tabular.html)]

### Changed

- Removed mlutils: Machine learning utility function package. A lot of its contents are not well-maintained, as a result, are already out-dated.
- LightGBM API walkthrough and a discussion about categorical features in tree-based models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/lightgbm.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/lightgbm.html)]
    - Upgrade LightGBM to 3.0.0, and deprecate out-dated content.
- Xgboost API walkthrough (includes hyperparameter tuning via scikit-learn like API). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/xgboost.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/xgboost.html)]
    - Upgrade XGBoost to 1.2.1, and deprecate out-dated content.

## [2020-09]

### Changed

- Probability Calibration for classification models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/prob_calibration.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html)]
    + Massive overhaul to the content. e.g. introducing two additional calibration methods, histogram binning and Plat Scaling Binning. Bundling all helper utility function in a package structure for ease of re-use.
- Multi-Label Text Classification with Fasttext and Huggingface Tokenizers. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/fasttext.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/fasttext.html)]
    + Update Huggingface Tokenizers to 0.8.1 API.

## [2020-06]

### Added

- Approximate Nearest Neighborhood Search with Navigable Small World. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/nsw.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/nsw.html)]

## [2020-05]

### Added

- Product Quantization for Model Compression. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/product_quantization.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/product_quantization.html)]
- Maximum Inner Product for Speeding Up Generating Recommendations. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/max_inner_product/max_inner_product.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/max_inner_product/max_inner_product.html)]

## [2020-04]

### Added

- Extremely Quick Guide to Unicode. [[markdown](https://github.com/ethen8181/machine-learning/blob/master/python/unicode.md)]
- MultiLabel Text Classification with Fasttext and Huggingface Tokenizers. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/fasttext.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/fasttext.html)]

### Changed

- FastAPI & Azure Kubernetes Cluster. End to end example of training a model and hosting it as a service. [[folder](https://github.com/ethen8181/machine-learning/blob/master/model_deployment/fastapi_kubernetes)]
    - Added application load testing with Apache Jmeter.

## [2020-03]

### Changed

- FastAPI & Azure Kubernetes Cluster. End to end example of training a model and hosting it as a service. [[folder](https://github.com/ethen8181/machine-learning/blob/master/model_deployment/fastapi_kubernetes)]
    - Added more best practices when specifying a deployment.

## [2020-02]

### Added

- FastAPI & Azure Kubernetes Cluster. End to end example of training a model and hosting it as a service. [[folder](https://github.com/ethen8181/machine-learning/blob/master/model_deployment/fastapi_kubernetes)]

### Changed

- Parallel programming with Python (threading, multiprocessing, concurrent.futures, joblib). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/parallel.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/parallel.html)]
    - Added a short section to asynchronous programming.
- Monotonic Constraint with Boosted Tree. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/monotonic.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/monotonic.html)]
    - The original notebook uses xgboost to demonstrate the feature. Added lightgbm example.
- Logging module. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/logging.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/logging.html)]
    - Added a section that emphasizes the importance of logging the full stack trace of an exception.

## [2020-01]

### Added

- [Kaggle: Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification/) Predicting insincere questions. [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_quora_insincere/)]

### Changed

- Seq2Seq for German to English Machine Translation - PyTorch. Includes quick intro to torchtext [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/1_torch_seq2seq_intro.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/1_torch_seq2seq_intro.html)]
    - Added more introduction to torchtext.

## [2019-12]

### Added

- Byte Pair Encoding (BPE) from scratch and quick walkthrough of sentencepiece. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/subword/bpe.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html)]
- Sentencepiece Subword tokenization for Text Classification. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/keras_subword_tokenization.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/text_classification/keras_subword_tokenization.html)]

### Changed

- Gaussian Mixture Model from scratch; AIC and BIC for choosing the number of Gaussians. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/GMM/GMM.ipynb)][[html](http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html)]
    - Fix erroneous log likelihood calculation.
    - Update deprecated function for plotting contour plots.

## [2019-11]

### Added

- Leveraging Pre-trained Word Embedding for Text Classification. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/keras_pretrained_embedding.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/text_classification/keras_pretrained_embedding.html)]
- Monotonic Constraint with Boosted Tree. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/monotonic.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/monotonic.html)]
- Probability Calibration for classification models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/prob_calibration.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html)]

## [2019-10]

### Added

- Seq2Seq with Attention for German to English Machine Translation - PyTorch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/2_torch_seq2seq_attention.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/2_torch_seq2seq_attention.html)]

## [2019-09]

### Added

- Seq2Seq with PyTorch for German to English Machine Translation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/1_torch_seq2seq_intro.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/1_torch_seq2seq_intro.html)]

## [2019-08]

### Added

- [Kaggle: Rossman Store Sales](https://www.kaggle.com/c/rossmann-store-sales/) Predicting daily store sales. Also introduces deep learning for tabular data. [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_rossman_store_sales/)]

### Changed

- Optimizing Pandas (e.g. reduce memory usage using category type). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pandas/pandas.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/pandas/pandas.html)]
	+ Added helper function to automatically determine optimal data type.
- Framing time series problem as supervised-learning. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/3_supervised_time_series.ipynb)][[html](http://ethen8181.github.io/machine-learning/time_series/3_supervised_time_series.html)]
	+ Added window-based features.

## [2019-06]

### Added

- Word2vec for Text Classification. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/word2vec_text_classification.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html)]

### Changed

- Word2vec (skipgram + negative sampling) using Gensim. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/word2vec/word2vec_detailed.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/word2vec/word2vec_detailed.html)]
	- Update to the more efficient file-based training.  

## [2019-04]

- Propensity Score Matching. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/causal_inference/matching.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/causal_inference/matching.html)]

## [2019-03]

### Added

- Short Walkthrough of PageRank. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/networkx/page_rank.ipynb)][[html](http://ethen8181.github.io/machine-learning/networkx/page_rank.html)]

## [2019-02]

### Added

- Quick Example of Factory Design Pattern. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/factory_pattern.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/factory_pattern.html)]
- Introduction to Multi-armed Bandits. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/reinforcement_learning/multi_armed_bandits.ipynb)][[html](http://ethen8181.github.io/machine-learning/reinforcement_learning/multi_armed_bandits.html)]

## [2019-01]

### Added

- Quantile Regression and its application in A/B testing.
  - Quick Introduction to Quantile Regression. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/quantile_regression/quantile_regression.ipynb.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.ipynb.html)]
  - Quantile Regression's application in A/B testing. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/quantile_regression/ab_test_regression.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/ab_test_regression.html)]

## [2018-12]

### Added

- First Foray Into Discrete/Fast Fourier Transformation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/fft/fft.ipynb)][[html](http://ethen8181.github.io/machine-learning/time_series/fft/fft.html)]

## [2018-11]

### Added

- Introduction to BM25 (Best Match). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/search/bm25_intro.ipynb.ipynb)][[html](http://ethen8181.github.io/machine-learning/search/bm25_intro.ipynb.html)]

## [2018-10]

### Added

- Kullback-Leibler (KL) Divergence. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/kl_divergence.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/kl_divergence.html)]
- Calibrated Recommendation for reducing bias/increasing diversity in recommendation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/calibration/calibrated_reco.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/calibration/calibrated_reco.html)]
- Influence Maximization from scratch. Includes discussion on Independent Cascade (IC), Submodular Optimization algorithms including Greedy and Lazy Greedy, a.k.a Cost Efficient Lazy Forward (CELF) [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/networkx/max_influence/max_influence.ipynb)][[html](http://ethen8181.github.io/machine-learning/networkx/max_influence/max_influence.html)]

## [2018-09]

### Added

Introduction to Residual Networks (ResNets) and Class Activation Maps (CAM). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/resnet_cam/resnet_cam.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/resnet_cam/resnet_cam.html)]

### Changed

Hosted html-version of all jupyter notebook on github pages.

## [2018-08]

### Added

- (Text) Content-Based Recommenders. Introducing Approximate Nearest Neighborhood (ANN) - Locality Sensitive Hashing (LSH) for cosine distance from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/content_based/lsh_text.ipynb)]
- Benchmarking ANN implementations (nmslib). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/ann_benchmarks/ann_benchmarks.ipynb)]

## [2018-07]

### Added

- Getting started with time series analysis with Exponential Smoothing (Holt-Winters). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/1_exponential_smoothing.ipynb)]
- Framing time series problem as supervised-learning. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/3_supervised_time_series.ipynb)]
- Tuning Spark Partitions. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/spark_partitions.ipynb)]

## [2018-06]

### Added

- Evaluation metrics for imbalanced dataset. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/imbalanced/imbalanced_metrics.ipynb)]

### Changed

- H2O API walkthrough (using GBM as an example). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/h2o/h2o_api_walkthrough.ipynb)]
    + Moved H2O notebook to its own sub-folder.
    + Added model interpretation using partial dependence plot.

## [2018-05]

### Added

- RNN, LSTM - PyTorch hello world. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/rnn/1_pytorch_rnn.ipynb)]
- Recurrent Neural Network (RNN) - language modeling basics. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/rnn_language_model_basic_keras.ipynb)]

## [2018-04]

### Added

- Long Short Term Memory (LSTM) - Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/rnn/2_tensorflow_lstm.ipynb)]
- Vanilla RNN - Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/rnn/1_tensorflow_rnn.ipynb)]
- WARP (Weighted Approximate-Rank Pairwise) Loss using lightfm. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/5_warp.ipynb)]

## [2018-03]

### Added

- Local Hadoop cluster installation on Mac. [[markdown](https://github.com/ethen8181/machine-learning/tree/master/big_data/local_hadoop.md)]
- Spark MLlib Binary Classification (using GBM as an example). [[raw zeppelin notebook](https://github.com/ethen8181/machine-learning/blob/master/big_data/sparkml/sparkml.json)][[Zepl](https://www.zepl.com/explore)]


## [2018-02]

### Added

- H2O API walkthrough (using GBM as an example). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/h2o/h2o_api_walkthrough.ipynb)]
- Factorization Machine from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/factorization_machine/factorization_machine.ipynb)]

### Changed

- The `spark` folder has been renamed to `big_data` to incorporate other big data tools.


## [2018-01]

### Added

- Partial Dependence Plot (PDP), model-agnostic approach for directional feature influence. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/partial_dependence/partial_dependence.ipynb)]
- Parallel programming with Python (threading, multiprocessing, concurrent.futures, joblib). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/parallel.ipynb)]

## [2017-12]

### Added

- LightGBM API walkthrough and a discussion about categorical features in tree-based models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/lightgbm.ipynb)]
- Curated tips and tricks for technical and soft skills. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/tips_and_tricks/tips_and_tricks.ipynb)]
- Detecting collinearity amongst features (Variance Inflation Factor for numeric features and Cramer's V statistics for categorical features), also introduces Linear Regression from a Maximum Likelihood perspective and the R-squared evaluation metric. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/collinearity.ipynb)]

### Changed

- Random Forest from scratch and Extra Trees. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/random_forest.ipynb)]
    - Refactored code for visualizating tree's feature importance.
- Building intuition on Ridge and Lasso regularization using scikit-learn. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/regularization/regularization.ipynb)]
    - Include section when there are collinear features in the dataset.
- mlutils: Machine learning utility function package [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/mlutils/)]
    - Refer to its changelog for details.
- data_science_is_software. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/data_science_is_software/notebooks/data_science_is_software.ipynb)]
    - Mention notebook extension, a project that contains various functionalities that makes jupyter notebook even more pleasant to work with.

## [2017-11]

### Added

- Introduction to Singular Value Decomposition (SVD), also known as Latent Semantic Analysis/Indexing (LSA/LSI).  [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/dim_reduct/svd.ipynb)]

## [2017-10]

### Added

- mlutils: Machine learning utility function package [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/mlutils/)]

### Changed

- Bernoulli and Multinomial Naive Bayes from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/naive_bayes/naive_bayes.ipynb)]
    - Fixed various typos and added a more efficient implementation of Multinomial Naive Bayes.
- TF-IDF (text frequency - inverse document frequency) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/tfidf/tfidf.ipynb)]
    - Moved to its own tfidf folder.
    - Included the full tfidf implementation from scratch.

### Changed

- Using built-in data structure and algorithm. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/1_data_structure.ipynb)]
    - Merged the content from the two notebooks `namedtuple and defaultdict` and `sorting with itemgetter and attrgetter` into this one and improved the section on priority queue.

## [2017-08]

### Added

- Understanding iterables, iterator and generators. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/iterator/iterator.ipynb)]
- Word2vec (skipgram + negative sampling) using Gensim (includes text preprocessing with spaCy). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/word2vec/word2vec_detailed.ipynb)]
- Frequentist A/B testing (includes a quick review of concepts such as p-value, confidence interval). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/frequentist_ab_test.ipynb)]
- AUC (Area under the ROC, precision/recall curve) from scratch (includes building a custom scikit-learn transformer). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/auc/auc.ipynb)]

### Changed

- Optimizing Pandas (e.g. reduce memory usage using category type). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pandas/pandas.ipynb)]
    - This is a revamp of the old content Pandas's category type.


## [2017-07]

### Added

- cohort : Cohort analysis. Visualize user retention by cohort with seaborn's heatmap and illustrating pandas's unstack. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/cohort/cohort.ipynb)

### Changed

- Bayesian Personalized Ranking (BPR) from scratch & AUC evaluation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/4_bpr.ipynb)]
    - A more efficient matrix operation using Hadamard product.
- Cython and Numba quickstart for high performance python. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/cython/cython.ipynb)
    - Added Numba parallel prange.
- ALS-WR for implicit feedback data from scratch & mean average precision at k (mapk) and normalized cumulative discounted gain (ndcg) evaluation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/2_implicit.ipynb)]
    - Included normalized cumulative discounted gain (ndcg) evaluation.
- Gradient Boosting Machine (GBM) from scratch. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/gbm/gbm.ipynb)
    - Added a made up number example on how GBM works.
- data_science_is_software. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/data_science_is_software/notebooks/data_science_is_software.ipynb)]
    - Mention `nbdime`, a tool that makes checking changes in jupyter notebook on github a lot easier.
    - Mention semantic versioning (what each number in the package version usually represents).
    - Mention `configparser`, a handy library for storing and loading configuration files.
- K-fold cross validation, grid/random search from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/model_selection.ipynb)]
    - Minor change in Kfolds educational implementation (original was passing redundant arguments to a method).
    - Minor change in random search educational implementation (did not realize scipy's .rvs method for generating random numbers returns a single element array instead of a number when you pass in size = 1).


## [2017-06]

This is the first time that the changelog file is added, thus every existing notebook will fall under the added category. Will try to group the log by month (one or two) in the future. Note that this repo will be geared towards Python3. Hence, even though the repo contains some R-related contents, they are not that well maintained and will most likely be translated to Python3. As always, any feedbacks are welcomed.

### Added

- Others (Genetic Algorithm)
- Regression (Linear, Ridge/Lasso)
- Market Basket Analysis (Apriori)
- Clustering (K-means++, Gaussian Mixture Model)
- Deep Learning (Feedforward, Convolutional Neural Nets)
- Model Selection (Cross Validation, Grid/Random Search)
- Dimensionality Reduction (Principal Component Analysis)
- Classification (Logistic, Bernoulli and Multinomial Naive Bayes)
- Text Analysis (TF-IDF, Chi-square feature selection, Latent Dirichlet Allocation)
- Tree Models (Decision Tree, Random Forest, Extra Trees, Gradient Boosting Machine)
- Recommendation System (Alternating Least Squares with Weighted Regularization, Bayesian Personalized Ranking)
- Python Programming (e.g. logging, unittest, decorators, pandas category type)
