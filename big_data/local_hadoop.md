# Setting up Hadoop Locally on Mac

Often times we wish to test or scripts locally before pushing our code to production machine and going through 2 factor authentication. This file documents the steps that have worked for me.


1. [Setup ssh to connect to localhost without a passphrase/password](https://stackoverflow.com/questions/7439563/how-to-ssh-to-localhost-without-password/10744443#10744443)

Make sure we enable `Remote Login` in `System Preference -> Sharing`.

```bash
# 1. Generate a key pair, note that if you already have a key pair,
# then you do not have to create a new one.

# we can check whether the key-pair exists
ls .ssh/ | grep id_rsa.pub

# if it gives an empty result, create the key-pair
ssh-keygen -t rsa
# Press enter for each line 

# append the contents to the authorized_keys file
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# then we should not be prompted to enter our password when ssh to localhost
ssh localhost
```

2. Download Hadoop and modify some configurations.

The following steps goes through setting up Hadoop in pseudo-distributed mode.

In this mode, Hadoop runs on a single node and each Hadoop daemon runs in a separate Java process.


- Download [Hadoop 2.8.1](http://www-eu.apache.org/dist/hadoop/common/hadoop-2.8.1/hadoop-2.8.1.tar.gz), you don't have to pick this version.
- Unpack the tar file and save it to our favorite location. In this case it will be `~/hadoop-2.8.1`.

Next we'll edit the some config files under `~/hadoop-2.8.1/etc/hadoop/`.


- `hdfs-site.xml`, HDFS's default backup file number is 3, we will change it to 1.

```
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
</configuration>
```

- `core-site.xml`, Configure HDFS's port number. 

```
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>
```

- `mapred-site.xml`, we can configure Hadoop to use Yarn as the resource manager framework.

```
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
</configuration>

```

- `yarn-site.xml`

```
<configuration>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
</configuration>
```


3. Format and start HDFS and Yarn, remember to change the <username> part to your user name.


```bash
# change hadoop path accordingly
cd ./hadoop-2.8.1

./bin/hdfs namenode -format
./sbin/start-dfs.sh
./bin/hdfs dfs -mkdir /user
./bin/hdfs dfs -mkdir /user/<username>
./sbin/start-yarn.sh

# use some hadoop file system commands to verify that its working
# create a temp directory, list it and remove it
./bin/hdfs dfs -mkdir temp
./bin/hdfs dfs -ls
./bin/hdfs dfs -rm -r temp
```

4. stop HDFS and Yarn after we are done.

```bash
./sbin/stop-yarn.sh
./sbin/stop-dfs.sh
```

5. None mandatory steps.

We can add `HADOOP_HOME` environment variable to `.bashrc` or `.bash_profile` for future use.

```bash
export HADOOP_HOME=~/hadoop-2.8.1
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin
```

In the future, we can start and close Hadoop under the root directory rather than going under the hadoop directory every time.

```bash
hdfs namenode -format
start-dfs.sh
hdfs dfs -mkdir /user
hdfs dfs -mkdir /user/<username>
start-yarn.sh

stop-yarn.sh
stop-dfs.sh
```

# Reference

- [Blog: Setting up Hadoop 2.6 on Mac OS X Yosemite](http://zhongyaonan.com/hadoop-tutorial/setting-up-hadoop-2-6-on-mac-osx-yosemite.html)
- [Blog: Setting up Hadoop 2.4 and Pig 0.12 on OSX locally](https://blueshift.com/setting-up-hadoop-2-4-and-pig-0-12-on-osx-locally/)
- [Blog: Apache Spark and Hadoop on a Macbook Air running OSX Sierra](https://medium.com/@jeremytarling/apache-spark-and-hadoop-on-a-macbook-air-running-osx-sierra-66bfbdb0b6f7)
- [Apache Hadoop Documentation: Hadoop: Setting up a Single Node Cluster](http://hadoop.apache.org/docs/r3.0.0/hadoop-project-dist/hadoop-common/SingleCluster.html)
