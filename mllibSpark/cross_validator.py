from pyspark import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, NaiveBayes, DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier, OneVsRest
#from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SQLContext, SparkSession
import time

def cross_validate(trainDataDF, testDataDF, estimator, paramGrid, evaluator, 
        numFolds=10, name_estimator="", bSave=False, filepath=""):

    print("***Starting %d-fold validation for estimator %s***" % (numFolds, name_estimator))
    crossval = CrossValidator(estimator=estimator,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator, 
                              numFolds=numFolds)

    start_time = time.time()
    model = crossval.fit(trainDataDF)
    print("Cross validation finished in %s seconds." % (time.time() - start_time))
    
    print("Avg cross-val metric %s" % model.avgMetrics)
    
    if bSave: 
        print("Saving to filepath %s." % filepath)
        model.write().overwrite().save(filepath)

    return model

def printEvaluationReport(prediction, evaluator):

    #first evaluation report
    accuracy = evaluator.evaluate(prediction, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(prediction, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(prediction, {evaluator.metricName: "weightedRecall"})
    f1 = evaluator.evaluate(prediction, {evaluator.metricName: "f1"})

    print("------Classification report-----")
    print("Accuracy: {0:.2%}".format(accuracy))
    print("Error rate: {0:.2%}".format(1.0 - accuracy))
    print("Weighted precision: {0:.2%}".format(precision))
    print("Weighted recall: {0:.2%}".format(recall))
    print("F1-score: {0:.2%}".format(f1))
    print()

if __name__ == "__main__":
    sc = SparkContext()

    spark = SparkSession\
            .builder\
            .appName("k-foldcv")\
            .getOrCreate()

    trainingData = MLUtils.loadLibSVMFile(sc, "paradiseXy_train.txt", multiclass=True)
    testData = MLUtils.loadLibSVMFile(sc, "paradiseXy_test.txt", multiclass=True)
    loadedDataDF_train = spark.createDataFrame(trainingData.map(lambda lp: (lp.label, lp.features.asML())), ['label', 'features'])
    loadedDataDF_test = spark.createDataFrame(testData.map(lambda lp: (lp.label, lp.features.asML())), ['label', 'features'])
    #print(loadedDataDF_train.show(truncate=False))
    

    # 1) Random Forest
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    #pipline = Pipeline(stages=[rf])

    paramGridRF = ParamGridBuilder() \
            .addGrid(rf.numTrees, [20]) \
            .build()

    evaluator = MulticlassClassificationEvaluator()

    modelRF = cross_validate(loadedDataDF_train, loadedDataDF_test, rf, paramGridRF, evaluator, 
            name_estimator="Random Forest", bSave=True, filepath="model/rf_model")
    predictionRF = modelRF.transform(loadedDataDF_test)

    printEvaluationReport(predictionRF, evaluator)


    # 2) Logistic Regression 
    lr = LogisticRegression(labelCol="label", featuresCol="features")

    paramGridLR = ParamGridBuilder() \
            .addGrid(lr.maxIter, [100]) \
            .build()

    modelLR = cross_validate(loadedDataDF_train, loadedDataDF_test, lr, paramGridLR, evaluator, 
            name_estimator="Logistic Regression", bSave=True, filepath="model/lr_model")
    predictionLR = modelLR.transform(loadedDataDF_test)

    printEvaluationReport(predictionLR, evaluator)

### ERROR: cannot work with negative features
    # 3) Multinomial Naive Bayes 
#    nb = NaiveBayes(labelCol="label", featuresCol="features", modelType="multinomial")
#
#    paramGridNB = ParamGridBuilder() \
#            .addGrid(nb.smoothing, [1.0]) \
#            .build()
#
#    modelNB = cross_validate(loadedDataDF_train, loadedDataDF_test, nb, paramGridNB, evaluator, 
#            name_estimator="Naive Bayes", bSave=False, filepath="model/nb_model")
#    predictionNB = modelNB.transform(loadedDataDF_test)
#
#    printEvaluationReport(predictionNB, evaluator)


    # 4) Decision Tree Classifier
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    paramGridDT = ParamGridBuilder() \
            .addGrid(dt.maxDepth, [5]) \
            .build()

    modelDT = cross_validate(loadedDataDF_train, loadedDataDF_test, dt, paramGridDT, evaluator, 
            name_estimator="Decision Tree", bSave=False, filepath="model/dt_model")
    predictionDT = modelDT.transform(loadedDataDF_test)

    printEvaluationReport(predictionDT, evaluator)


    # 5) Linear SVM One-Vs-Rest
    svm = LinearSVC(labelCol="label", featuresCol="features")
    ovr = OneVsRest(classifier=svm, parallelism=16)

    paramGridSVM = ParamGridBuilder() \
            .addGrid(svm.regParam, [0.0]) \
            .build()

    modelSVM = cross_validate(loadedDataDF_train, loadedDataDF_test, ovr, paramGridSVM, evaluator, 
            name_estimator="Linear SVM One-vs-Rest", bSave=True, filepath="model/svm_model")
    predictionSVM = modelSVM.transform(loadedDataDF_test)

    printEvaluationReport(predictionSVM, evaluator)

    # 5) ANN - Multi-layer Perceptron Classifier
    mlp = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=[28, 100, 100, 9],  maxIter=200)

    evaluator = MulticlassClassificationEvaluator()

    paramGridMLP = ParamGridBuilder() \
            .addGrid(mlp.stepSize, [0.03]) \
            .build()

    modelMLP = cross_validate(loadedDataDF_train, loadedDataDF_test, mlp, paramGridMLP, evaluator, 
            name_estimator="Multi-Layer Perceptron", bSave=True, filepath="model/mlp_model")
    predictionMLP = modelMLP.transform(loadedDataDF_test)

    printEvaluationReport(predictionMLP, evaluator)




    # get these metrics later
    exit()    
    #set up rdd for prediction labels
    rf_bestModel = model.bestModel #get the best RF model from cross validation
    predictionAndLabels = testData.map(lambda lp: (float(rf_bestModel.predict(lp.features)), lp.label))

    rdd2 = sc.parallelize(predictionAndLabels)

    #second evaluation report
    #metrics = MulticlassMetrics(predictionAndLabels)
    metrics = MulticlassMetrics(rdd2)
    #overall stat
    confMat = metrics.confusionMatrix().toArray()
    acc = metrics.accuracy
    prec = metrics.precision()
    reca = metrics.recall()
    f1Score = metrics.fMeasure()
    
    print("Confusion Matrix")
    print(confMat)

    print("Summary Stats")
    print("Accuracy = %s" % acc)
    print("Precision = %s" % prec)
    print("Recall = %s" % reca)
    print("F1-score = %s" % f1Score)

    print() 


    #stats by class
    labels = loadedDataDF_test.map(lambda lp: lp.label).distinct().collect()
    for label in sorted(labels):
        print("Class %s precision = %s" % (label, metrics.precision(label)))
        print("Class %s recall = %s" % (label, metrics.recall(label)))
        print("Class %s F1 Measure= %s" % (label, metrics.fMeasure(label)))
        print("Class %s FPR = %s" % (label, metrics.falsePositiveRate(label)))
        print("Class %s TPR = %s" % (label, metrics.truePositiveRate(label)))

    print()

    #weighted stats
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted F(1) score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted FPR = %s" % metrics.weightedFalsePositiveRate)
    print("Weighted TPR = %s" % metrics.weightedTruePositiveRate)
