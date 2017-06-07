## Pyspark Installation On Mac

To run spark in the notebook. First download `pyspark` from anaconda (if you're downloading the latest anaconda it might be already included).

```shell
conda install -y -c anaconda-cluster spark
```

Download the latest version of Apache Spark from this [link](https://spark.apache.org/downloads.html). After downloading it (sticking with the default configuration should be fine if we're just experimenting with it on our local machine), unzip it and place the folder in our home directory and change the folder name to just `spark`. Before starting the notebook, we will need to define these environment variables 

```shell
export SPARK_HOME=~/spark
export PYSPARK_PYTHON=python3
export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
```

On Unix/Mac, this can be done in .bashrc or .bash_profile.