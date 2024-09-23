import common
from classification import split_train_test, train_model, model_predict, evaluate_model
import pandas as pd
import pickle
from pyspark.sql import SparkSession
import pyspark

spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()

def main(data, target_col, category):
    #df = common.load_file(data)
    df = spark.read.csv(data, header=True, inferSchema=True)
    preprocessed_df, gender = common.preprocessing_data(df)

    X_train, X_test, y_train, y_test = split_train_test(preprocessed_df, target_col)

    rf_model, kn_model, logi_model, svc_model, nb_model, dt_model, xg_model, stacking_model, voting_model = train_model(X_train, y_train)

    models = [
    (rf_model, "Random Forest"),
    #(kn_model, "K Neighbors"),
    (logi_model, "Logistic Regression"),
    #(svc_model, "SVC"),
    (nb_model, "Naive Bayes"),
    (dt_model, "Decision Tree"),
    (xg_model, "XGB"),
    #(stacking_model, "Stacking"),
    #(voting_model, "Voting")
    ]

    print("--- Results of evaluate of each model ---")
    for model, name in models:
        pred = model_predict(model, X_test)
        eval = evaluate_model(y_test, pred)
        print(f"\n{name}:\n", eval)

        with open(f"{name}_{category}.pkl", 'wb') as file:
            pickle.dump(model, file)

    # rf_pred = model_predict(rf_model, X_test)
    # kn_pred = model_predict(kn_model, X_test)
    # logi_pred = model_predict(logi_model, X_test)
    # svc_pred = model_predict(svc_model, X_test)
    # nb_pred = model_predict(nb_model, X_test)
    # dt_pred = model_predict(dt_model, X_test)
    # xg_pred = model_predict(xg_model, X_test)
    # stacking_pred = model_predict(stacking_model, X_test)
    # voting_pred = model_predict(voting_model, X_test)

    # rf_eval = evaluate_model(y_test, rf_pred)
    # kn_eval = evaluate_model(y_test, kn_pred)
    # logi_eval = evaluate_model(y_test, logi_pred)
    # svc_eval = evaluate_model(y_test, svc_pred)
    # nb_eval = evaluate_model(y_test, nb_pred)
    # dt_eval = evaluate_model(y_test, dt_pred)
    # xg_eval = evaluate_model(y_test, xg_pred)
    # stacking_eval = evaluate_model(y_test, stacking_pred)
    # voting_eval = evaluate_model(y_test, voting_pred)

    # print("--- Result of evaluation each model ---\n")
    # print("\nRandom Forest:\n", rf_eval)
    # print("\nK Neighbors:\n", kn_eval)
    # print("\nLogistic Regression:\n", logi_eval)
    # print("\nSVC:\n", svc_eval)
    # print("\nNaive Bayes:\n", nb_eval)
    # print("\nDecision Tree:\n", dt_eval)
    # print("\nXGB:\n", xg_eval)
    # print("\nStacking:\n", stacking_eval)
    # print("\nVoting:\n", voting_eval)
    
main('./data/heart_failure_clinical_records_dataset.csv', 'DEATH_EVENT', 'Medical')
