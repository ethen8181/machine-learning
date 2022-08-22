# Speeding Up Transformers Inferencing

Transfer learning on large scale pre-trained models based on transformation architecture on downstream applications have been gaining a lot of popularity due to its promising gains on model performance. However, in real world applications, often times, we have other constraints around latency, throughout as well as memory. This folder show cases several techniques that are commonly used to speed up transformers model's inferencing while still retaining the majority of its performance.

- Finetuning Pre-trained BERT Model on Text Classification Task And Inferencing with ONNX Runtime. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_deployment/transformers/text_classification_onnxruntime.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_deployment/transformers/text_classification_onnxruntime.html)]



# Reference

- [Blog: How We Scaled Bert To Serve 1+ Billion Daily Requests on CPUs](https://medium.com/@quocnle/how-we-scaled-bert-to-serve-1-billion-daily-requests-on-cpus-d99be090db26)