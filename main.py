import pandas as pd


for i in range(1):

    instancesNumber = 200000
    env="yelp"

    if env=="twitter":

        df_ = pd.read_csv("datasets/TwitterSentiment140_Shuffled.csv", names=["target", "id", "date", "flag", "user", "text"], encoding='latin-1')
        #df_.sample(frac=1).to_csv("TwitterSentiment140_Shuffled.csv", index=False)
        df_[["text", "target"]].sample(instancesNumber).to_csv("TwitterSentiment140_Shuffled.csv", index=False)

    elif env=="yelp":
        #uncomment if using twitter
        df_ = pd.read_csv("datasets/yelp_review_clean.csv", encoding='latin-1')
        ##df_.sample(frac=1).to_csv("TwitterSentiment140_Shuffled.csv", index=False)
        df_[["text", "stars"]].sample(instancesNumber).to_csv("yelp_review_clean.csv", index=False)
        
    from helper_functions import loadDatasetTey

    from river import naive_bayes, metrics, stream
    import sys
    import re
    import time
    import gc

    run_ht = False
    run_w2v = False
    run_bert = False
    run_iwc = True


    totalInstances = 0
    cache_key = f"twitter{time.time()}"

    """
    if "--totalInstances" in sys.argv:
        argumentIndex = sys.argv.index("--totalInstances")
        totalInstances = int(sys.argv[argumentIndex + 1])
        print(totalInstances)
    if "-h" in sys.argv:
        print("Usage: \n\t--totalInstances : sets the number of instaces to be used on testing")
        exit()
    """
    print("It started")

    # Feature extractors




    # print(testFeatureExtractor([ht, word2Tey], [modelHT, modelW2V], [m    etricHT, metricW2V], dataset))

    print(f"Start loading {env} dataset")

    dataset = loadDatasetTey(env=env)
    #dataset = loadDatasetTey()
    cache = stream.Cache()

    print("Starting test routine for", totalInstances, "intances")
    log = []

    if run_ht:
        from hashing_trick import HashingTrickTey

        ht = HashingTrickTey()
        modelHT = naive_bayes.GaussianNB()
        metricHT = metrics.Accuracy()

        cont = 0
        start = 0

        for instance, label in cache(dataset, key=cache_key):
            # Timer is started ghere so the time for loading the
            # dataset is not considered
            
            if cont == 0:
                cont += 1
                # print(instance)
                start = time.time()

            # Retrieve instance's textual parameter
            # It is expected the instance to be a dictionary and 
            # the first parameter (and only one) to be a text
            text_parameter = instance[list(instance.keys())[0]]

            # Removes special characters from text
            text_parameter = re.sub('\W+', ' ', text_parameter)

            text_parameter = text_parameter.lower()

            ht.fit(text_parameter)
            extracted_features = ht.transform_one(text_parameter)

            probs = modelHT.predict_proba_one(extracted_features)
            
            # {1: %, 2:%}
            # {"4", "5"}
            if len(probs) > 0:
                y_pred = max(probs, key=lambda k: probs[k])
            else:
                y_pred = 0

            modelHT.learn_one(extracted_features, label)
            metricHT.update(label, y_pred)
        
            if totalInstances != 0 and cont > totalInstances:
                print(f"{totalInstances} -- {cont}")
                break

            if (totalInstances != 0 and cont % (totalInstances/10) == 0) or (totalInstances == 0 and cont % 10000 == 0):
                print("\t", cont, "of", totalInstances, "instances processed")  
                log += f"\n{cont} of {totalInstances} instances processed. {time.time()-start} seconds elapsed."
            

            """
            if totalInstances != 0 or cont % 10000 == 0:
                if totalInstances != 0 and cont > totalInstances:         
                    break
                if  cont % 10000 == 0 or (totalInstances != 0 and cont % (totalInstances/10) == 0):
                    # print("y_pred", y_pred, "label", label)
                    print("\t", cont, "of", totalInstances, "instances processed")
                    with open("ht.txt", 'a') as f:
                        f.write(f"\n{cont} of {totalInstances} instances processed. {time.time()-start} seconds elapsed.")
            """
            cont += 1

        with open("ht.txt", 'a') as f:
            f.write("".join(log))
        print("Hashing Trick", metricHT, "Time elapsed (sec):", time.time() - start)

    gc.collect()
    if run_w2v:
        from word2vec import Word2VecTey

        word2Tey = Word2VecTey(size=100)
        modelW2V = naive_bayes.GaussianNB()
        metricW2V = metrics.Accuracy()
        
        cont = 0
        start = 0
        for instance, label in cache(dataset, key=cache_key):
            # We start timer here so the time for loading the
            # dataset is not considered
            if cont == 0:
                cont += 1
                # print(instance)
                start = time.time()

            # Retrieve instance's textual parameter
            # It is expected the instance to be a dictionary and 
            # the first parameter (and only one) to be a text
            text_parameter = instance[list(instance.keys())[0]]

            # Removes special characters from text
            text_parameter = re.sub('\W+', ' ', text_parameter)

            text_parameter = text_parameter.lower()

            word2Tey.fit(text_parameter)
            extracted_features = word2Tey.transform_one(text_parameter)

            probs = modelW2V.predict_proba_one(extracted_features)

            if len(probs) > 0:
                y_pred = max(probs, key=lambda k: probs[k])
            else:
                y_pred = 0

            modelW2V.learn_one(extracted_features, label)
            metricW2V.update(label, y_pred)


            if totalInstances != 0 and cont > totalInstances:
                print(f"{totalInstances} -- {cont}")
                break

            if (totalInstances != 0 and cont % (totalInstances/10) == 0) or (totalInstances == 0 and cont % 10000 == 0):
                print("\t", cont, "of", totalInstances, "instances processed")  
                log += f"\n{cont} of {totalInstances} instances processed. {time.time()-start} seconds elapsed."
            
            cont += 1

        with open("w2v.txt", 'a') as f:
            f.write("".join(log))
            f.write(f"Word2Vec {metricW2V} Time elapsed (s): {time.time() - start}")
        print("Word2Vec", metricW2V, "Time elapsed (s):", time.time() - start)

    gc.collect()
    if run_bert:
        from bert import iwc
        
        bertey = BertEy()
        modelBert = naive_bayes.GaussianNB()
        metricBert = metrics.Accuracy()
        iwc = IncrementalWordContext(10000, 116, 3, False)

        cont = 0
        start = 0
        for instance, label in cache(dataset, key=cache_key):
            # We start timer here so the time for loading the
            # dataset is not considered
            if cont % 100 == 0 and cont != 0: print(f"cont {cont} - {metricBert.get()}")
            if cont == 0:
                cont += 1
                # print(instance)
                start = time.time()

            # Retrieve instance's textual parameter
            # It is expected the instance to be a dictionary and 
            # the first parameter (and only one) to be a text
            text_parameter = instance[list(instance.keys())[0]]

            # Removes special characters from text
            text_parameter = re.sub('\W+', ' ', text_parameter)

            text_parameter = text_parameter.lower()

            bertey.fit(text_parameter)
            extracted_features = bertey.transform_one(text_parameter)
            
            iwc.fit(text_parameter)
            extracted_features_ = iwc.transform_one(text_parameter, for_river=True)

            extracted_features = {k: v for k, v in enumerate(list(extracted_features.values()) + list(extracted_features_.values()))}

            probs = modelBert.predict_proba_one(extracted_features)

            if len(probs) > 0:
                y_pred = max(probs, key=lambda k: probs[k])
            else:
                y_pred = 0

            modelBert.learn_one(extracted_features, label)
            metricBert.update(label, y_pred)


            if totalInstances != 0 and cont > totalInstances:
                print(f"{totalInstances} -- {cont}")
                break

            if (totalInstances != 0 and cont % (totalInstances/10) == 0) or (totalInstances == 0 and cont % 10000 == 0):
                print("\t", cont, "of", totalInstances, "instances processed")  
                log += f"\n{cont} of {totalInstances} instances processed. {time.time()-start} seconds elapsed."
            
            cont += 1

        with open("bert.txt", 'a') as f:
            f.write("".join(log))
            f.write(f"BERT {metricBert.get()} - Time elapsed (s): {time.time() - start}")
        print("BERT", metricBert, "Time elapsed (s):", time.time() - start)

    gc.collect()
    if run_iwc:
        from incremental_word_context import IncrementalWordContext
        
        #bertey = BertEy()
        modelIwc = naive_bayes.GaussianNB()
        metricIwc = metrics.Accuracy()
        iwc = IncrementalWordContext(10000, 500, 7, True)

        cont = 0
        start = 0
        for instance, label in cache(dataset, key=cache_key):
            # We start timer here so the time for loading the
            # dataset is not considered
            if cont % 1000 == 0 and cont != 0: print(f"cont {cont} - {metricIwc.get()}")
            if cont == 0:
                cont += 1
                # print(instance)
                start = time.time()

            # Retrieve instance's textual parameter
            # It is expected the instance to be a dictionary and 
            # the first parameter (and only one) to be a text
            text_parameter = instance[list(instance.keys())[0]]

            # Removes special characters from text
            text_parameter = re.sub('\W+', ' ', text_parameter)

            text_parameter = text_parameter.lower()
            
            iwc.fit(text_parameter)
            extracted_features = iwc.transform_one(text_parameter, for_river=True)

            #extracted_features = {k: v for k, v in enumerate(list(extracted_features_))}

            probs = modelIwc.predict_proba_one(extracted_features)

            if len(probs) > 0:
                y_pred = max(probs, key=lambda k: probs[k])
            else:
                y_pred = 0

            modelIwc.learn_one(extracted_features, label)
            metricIwc.update(label, y_pred)


            if totalInstances != 0 and cont > totalInstances:
                print(f"{totalInstances} -- {cont}")
                break

            if (totalInstances != 0 and cont % (totalInstances/10) == 0) or (totalInstances == 0 and cont % 10000 == 0):
                print("\t", cont, "of", totalInstances, "instances processed")  
                log += f"\n{cont} of {totalInstances} instances processed. {time.time()-start} seconds elapsed."
            
            cont += 1

        with open("iwc.txt", 'a') as f:
            f.write("".join(log))
            f.write(f"IWC {metricIwc.get()} - Time elapsed (s): {time.time() - start}")
        print("IWC", metricIwc, "Time elapsed (s):", time.time() - start)

    cache.clear_all()