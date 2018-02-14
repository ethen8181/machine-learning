## Pyspark Installation On Mac

To run spark in the notebook. First download [java](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) if you haven't already, then install `pyspark` from Anaconda or pypi.

```shell
# https://anaconda.org/conda-forge/pyspark
conda install -c conda-forge pyspark

#  pypi
pip install pyspark
```

Download the latest version of Apache Spark from this [link](https://spark.apache.org/downloads.html). After downloading it (sticking with the default configuration should be fine if we're just experimenting with it on our local machine), unzip it and place the folder in our home directory and change the folder name to just `spark`. Before starting the notebook, we will need to define these environment variables. On Unix/Mac, this can be done in .bashrc or .bash_profile.

```shell
# our spark home matches the folder name `spark`
export SPARK_HOME=~/spark

# set this to the python version you're running
# the following link contains instructions to find your python path
# if you're using anaconda
# https://docs.anaconda.com/anaconda/user-guide/tasks/integration/python-path
export PYSPARK_PYTHON=~/anaconda3/bin/python
```

After that we can try lauching the spark console

```
cd spark

# launching pyspark
./bin/pyspark

# or to launch the scala console
./bin/spark-shell

```