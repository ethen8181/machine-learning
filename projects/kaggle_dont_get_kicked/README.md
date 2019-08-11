# Kaggle Don't Get Kicked

Predict if a car purchased at auction is a unfortunate purchase. Problem description is available at https://www.kaggle.com/c/DontGetKicked

## Installation

This assumes the user already has Anaconda installed and is targeted for Python3.5

```bash
pip install -r requirements.txt
```

And install the [`mlutils`](https://github.com/ethen8181/machine-learning/tree/master/projects/mlutils) package which contains customized utility function.

```bash
# navigate to the directory
cd ../mlutils

# install the package
python setup.py install
```

For the modelling part, the [XGBoost](https://github.com/dmlc/xgboost) library is used and based on experience it's best to install it from source. The following section contains the installation instruction for Mac.

```bash
# install gcc from brew, 
# note that the second command can 
# take up to 30 minutes, be patient
brew tap homebrew/versions
brew install gcc --without-multilib

# install xgboost
git clone --recursive https://github.com/dmlc/xgboost 
cd xgboost

# open make/config.mk and uncomment these two lines

# export CC = gcc
# export CXX = g++

# but depending on the installation, we may need to change
# the two lines above to

# export CC = gcc-7
# export CXX = g++-7

# we can check the number after the "-" by entering 
# brew install gcc --without-multilib
# it will warn us the version if we already have it

# start the build
cp make/config.mk .
make -j4

cd python-package 
sudo python setup.py install
```

## Usage

```bash
# assuming you're at the project's root directory

# train the model on the training set and store it
python src/main.py --train --inputfile training.csv --outputfile prediction.csv

# predict on future dataset and output the prediction
# to a .csv file in a output directory (will be created
# one level above where the script is if it doesn't exist yet)
python src/main.py --inputfile test.csv --outputfile prediction_future.csv
```

## Documentation

- `src/main.ipynb` Jupyter Notebook that contains a walkthrough of the overall process. This is the best place to start. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/projects/kaggle_dont_get_kicked/src/main.ipynb)][[html](http://ethen8181.github.io/machine-learning/projects/kaggle_dont_get_kicked/src/main.html)]
- `src/main.py` Once you're acquainted with the process, you can just run this Python script to run the end to end pipeline. [[Python script](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_dont_get_kicked/src/main.py)]
- `src/utils.py` Utility function for the project used throughout the Jupyter notebook and Python script. [[Python script](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_dont_get_kicked/src/utils.py)]
