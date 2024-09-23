import pandas as pd
#import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, NaiveBayes, DecisionTreeClassifier
from xgboost.spark import SparkXGBClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def split_train_test(data, target_col, train_ratio=0.8, seed=42):
    if target_col not in data.columns:
        raise KeyError(f"Target column, 'f{target_col}', is not in your dataset.")
    
    X = data.drop(target_col)
    y = data.select(target_col)

    # add index column
    w = Window.orderBy(F.lit(1))
    X_indexed = X.withColumn("index", F.row_number().over(w))
    y_indexed = y.withColumn("index", F.row_number().over(w))

    # Split train and test set
    train, test = X_indexed.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

    # Index extraction of train and test sets
    train_indices = train.select("index")
    test_indices = test.select("index")

    #  Split X and y into train and test sets
    X_train = train.drop("index")
    X_test = test.drop("index")
    y_train = y_indexed.join(train_indices, on="index", how="inner").drop("index")
    y_test = y_indexed.join(test_indices, on="index", how="inner").drop("index")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    X_train_pd = X_train.toPandas()
    y_train_pd = y_train.toPandas()

    feature_cols = X_train.columns
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    data = X_train.join(y_train, on=X_train.index == y_train.index).drop("index")

    rf = RandomForestClassifier(labelCol=y_train, featuresCol="features", numTrees=100)
    logi = LogisticRegression(labelCol=y_train, featuresCol='features')
    nb = NaiveBayes(labelCol=y_train, featuresCol='features', smoothing=1.0, modelType="multinomial")
    dt = DecisionTreeClassifier(labelCol=y_train, featuresCol='features')
    xgb = SparkXGBClassifier(labelCol=y_train, featuresCol="features", numWorkers=4)

     estimators = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('Naive', GaussianNB()),
        ('SVC', SVC(kernel='linear', C=1.0, probability=True)),
        ('DecisionTree', DecisionTreeClassifier())
    ]

    stacking_model = StackingClassifier(estimators=estimators, final_estimator=XGBClassifier())
    voting_model = VotingClassifier(estimators=estimators, voting='hard')

    rf_pipeline = Pipeline(stages=[assembler, rf])
    logi_pipeline = Pipeline(stages=[assembler, logi])
    nb_pipeline = Pipeline(stages=[assembler, nb])
    dt_pipeline = Pipeline(stages=[assembler, dt])
    xgb_pipeline = Pipeline(stages=[assembler, xgb])

    rf_model = rf_pipeline.fit(data)
    logi_model = logi_pipeline.fit(data)
    nb_model = nb_pipeline.fit(data)
    dt_model = dt_pipeline.fit(data)
    xgb_model = xgb_pipeline.fit(data)
    stacking_model.fit(X_train, y_train)
    voting_model.fit(X_train, y_train)

    return rf_model, logi_model, nb_model, dt_model, xgb_model, stacking_model, voting_model

def model_predict(model, X_test):
    return model.predict(X_test)

def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    evaluation = {
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "classification_report": report
    }
    return evaluation
