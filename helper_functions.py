import river
import pandas as pd
import sys
import os
from sklearn.datasets import fetch_20newsgroups
from timeit import default_timer as timer
from datetime import timedelta
import ssl
import re

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

def loadDatasetTey(env="SMSSpam"):
    if env == "SMSSpam":
        # Returns river textual stream dataset for binary classification
        return river.datasets.SMSSpam()

    elif env == "newsGroups":
        # Returns sklearn news textual database as a river stream
        newsgroups = fetch_20newsgroups(subset='train')

        X = pd.DataFrame({"text": newsgroups.data,
                          "target": newsgroups.target})
        # print(X.info)

        y = X.pop("target")

        return river.stream.iter_pandas(X=X, y=y)

    elif env == "yelp":
        return river.stream.iter_csv("yelp_review_clean.csv", target="stars")
    
    elif env == "twitter":
        return river.stream.iter_csv("TwitterSentiment140_Shuffled.csv", target="target")

    else:
        exit()
        print("please enter valid dataset name")

hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)


def testFeatureExtractor(featureExtractors, models, metrics, data_stream_iterator):
    dictTey = {}
    for i in range(len(featureExtractors)):
        start = timer()
        cont = 0
        for instance, label in data_stream_iterator:
            
            # Retrieve instance's textual parameter
            # It is expected the instance to be a dictionary and 
            # the first parameter (and only one) to be a text
            text_parameter = instance[list(instance.keys())[0]]

            # Removes special characters from text
            text_parameter = re.sub('\W+',' ', text_parameter)
            
            text_parameter = text_parameter.lower()

            featureExtractors[i].fit(text_parameter)

            featureEx = featureExtractors[i].transform_one(document=text_parameter)
            probs = models[i].predict_proba_one(featureEx)

            if len(probs) > 0:
                y_pred = max(probs, key=lambda k: probs[k])
            else:
                y_pred = 0

            models[i].learn_one(featureEx, label)
            metrics[i].update(label, y_pred)

            if cont > 50:
                break
            else:
                cont += 1
            # Dataset iterator end

        end = timer()
        dictTey[i] = [metrics[i], timedelta(seconds=end-start)]

        # Feature Extrators iterator end

    print(dictTey)
    return pd.DataFrame(data=dictTey.values(),
                        columns=["accuracy", "time elapsed"],
                        index=[type(i) for i in featureExtractors])