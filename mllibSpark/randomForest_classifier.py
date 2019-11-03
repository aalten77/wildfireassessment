from pyspark import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel

if __name__ == "__main__":
    sc = SparkContext(appName="RandomForestClassifier")

    trainingData = MLUtils.loadLibSVMFile(sc, "paradiseXy_train.txt", multiclass=True)
    testData = MLUtils.loadLibSVMFile(sc, "paradiseXy_test.txt", multiclass=True)
    
    #(trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    model = RandomForest.trainClassifier(trainingData, numClasses=9, categoricalFeaturesInfo={}, 
            numTrees=3, featureSubsetStrategy="auto", 
            impurity='gini', maxDepth=4, maxBins=32)

    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
            lambda lp: lp[0] != lp[1]).count() / float(testData.count())
    print("Test Error = " + str(testErr))
    print("Learned classification forest model:")
    print(model.toDebugString())

    model.save(sc, "target/tmp/myRandomForestClassificationModel2")
    sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel2")
