# Movie Predictor

This is a demonstration of using an SVM algorithm to separate movies into two groups (like vs dislike). The user must first train the model by manipulating the `movies_metadata.csv` file in the data directory. The first column of this data set is used for training and prediction data. Place a one "1" in this column for any movie you liked and a zero "0" for any you did not. Place a "p" in this column for any movie that you would like the program to predict for you based on your previous choices. The results aren't great but they are adequate for a demonstration.

The more movies you categorize, the better the results are likely to be. You must categorize at least two movies you like and two that you do not for there to be any results. You must also ask for at least one title to be predicted.

The data files are modified versions of the Kaggle movies dataset [https://www.kaggle.com/rounakbanik/the-movies-dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) with modification to make the data easier to work with in Python and remove some anomalous data.

## Installation

**N.B.** Data files are only contained within the releases but are not in the main repo.

You must have previously installed `pip` and `virtualenv` and be able to use them as your user.

For details on installing `pip` see: [https://pip.pypa.io/en/stable/installing/](https://pip.pypa.io/en/stable/installing/)

For details on installing `virtualenv` see:
[https://virtualenv.pypa.io/en/stable/installation/](https://virtualenv.pypa.io/en/stable/installation/)

  1. Download the latest release from the releases page. (Releases have the data files but the repo does not).
  1. Open the directory via the command line.
  1. Activate `virtualenv` via the following command: `virtualenv .`
  1. Install requirements via the following command: `pip install -r requirements.txt`
  1. Edit the csv data file and then run `python3 predict.py`

### Example Installation...

```bast
unzip movie-predictor.zip
cd movie-predictor
virtualenv .
source ./bin/activate
pip install -r requirements.txt
python3 predict.py
```
