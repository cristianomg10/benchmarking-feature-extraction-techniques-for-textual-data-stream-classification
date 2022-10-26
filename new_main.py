import pandas as pd
from helper_functions import loadDatasetTey
from river import naive_bayes, metrics, stream
import sys
import re
import time
from datetime import datetime
import gc

# Models to be tested
from hashing_trick import HashingTrickTey
from word2vec import Word2VecTey
from bert import BertEy
from incremental_word_context import IncrementalWordContext

number_executions = 2
datasets_available = ["twitter", "yelp"] #, "imdb"]
instances_test = [10000] #, 20000, 30000, 50000, 100000, 200000]
dimensions = [100] #, 384, 500]
store_kappa_for_checking = False

filename = datetime.now().strftime("%Y%m%d%H%M%S-results.csv")
with open(f"{filename}", 'a') as f:
    f.write("execution,method,dimension,dataset,dataset_size,accuracy,kappa,time\n")

for dimension in dimensions:
    models = [
        ('bert', BertEy()),
        ('hashing-tricks', HashingTrickTey(dimension)),
        ('iwc', IncrementalWordContext(10000, dimension, 7, True)),
        ('word2vec', Word2VecTey(size=dimension)),
    ]

    for env in datasets_available:
        for instancesNumber in instances_test:
            for i in range(number_executions):
                print(f"Starting iteration {i}")
                if env == "twitter":
                    df_ = pd.read_csv("datasets/TwitterSentiment140_Shuffled.csv", names=["target", "id", "date", "flag", "user", "text"], encoding='latin-1')
                    df_[["text", "target"]].sample(instancesNumber).to_csv("TwitterSentiment140_Shuffled.csv", index=False)

                elif env == "yelp":
                    df_ = pd.read_csv("datasets/yelp_review_clean.csv", encoding='latin-1')
                    df_[["text", "stars"]].sample(instancesNumber).to_csv("yelp_review_clean.csv", index=False)
                    
                totalInstances = 0
                cache_key = f"{env}{time.time()}"

                print("It started")
                print(f"Start loading {env} dataset")

                dataset = loadDatasetTey(env=env)
                cache = stream.Cache()

                print("Starting test routine for", totalInstances, "instances")
                log = []

                for text_model_name, text_model in models:  
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting {text_model_name} ({dimension})")
                    ml_model = naive_bayes.GaussianNB()
                    metric = metrics.Accuracy()
                    metric_1 = metrics.CohenKappa()

                    cont = 0
                    start = 0

                    for instance, label in cache(dataset, key=cache_key):
                        
                        if cont == 0:
                            cont += 1
                            start = time.time()

                        text_parameter = instance[list(instance.keys())[0]]
                        text_parameter = re.sub('\W+', ' ', text_parameter)
                        text_parameter = text_parameter.lower()

                        text_model.fit(text_parameter)

                        extracted_features = text_model.transform_one(text_parameter)
                        probs = ml_model.predict_proba_one(extracted_features)

                        if len(probs) > 0:
                            y_pred = max(probs, key=lambda k: probs[k])
                        else:
                            y_pred = 0

                        ml_model.learn_one(extracted_features, label)

                        metric.update(label, y_pred)
                        metric_1.update(label, y_pred)
                        if store_kappa_for_checking == True:
                            with open(f"{text_model_name}_{dimension}_{env}_{instancesNumber}_{i}-kappa-check.csv", "a") as f:
                                f.write(f"{label},{y_pred}\n")
                        
                        if (totalInstances != 0 and cont % (totalInstances/10) == 0) or (totalInstances == 0 and cont % 10000 == 0):
                            print("\t", cont, "of", totalInstances, "instances processed")                     
                        cont += 1

                    with open(f"{filename}", 'a') as f:
                        f.write(f"{i},{text_model_name},{dimension},{env},{instancesNumber},{metric.get()},{metric_1.get()},{time.time() - start}\n")
                    print(f"{text_model_name} - Accuracy: {metric} Time elapsed (sec): {time.time() - start}")

                    gc.collect()

                try:
                    cache.clear_all()
                except:
                    pass