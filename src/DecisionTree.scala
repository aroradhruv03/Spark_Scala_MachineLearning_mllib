import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo.Classification
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.{SparkConf, SparkContext}
/**
  * Created by dhruv on 4/7/16.
  */
object DecisionTree {

  val INP_FILE = "glass.data"
  val INP_FILE_MOVIES = "movies.dat"
  val OUT_DIR = "D:\\BigData\\Assign3\\"

  // Function to clean the data
  def parseLine(line: String): LabeledPoint = {
    val cleanLine = line.replaceAll(","," ")
    val parts = cleanLine.split(" ")
    LabeledPoint( parts(parts.length-1).toDouble-1, Vectors.dense(parts.slice(1,parts.length-1).map(_.toDouble)) )
  }

  def main (args: Array[String])
  {
    val conf = new SparkConf().setAppName("Decision Tree Using Spark").setMaster("local")
    val sc = new SparkContext(conf)


    val file = sc.textFile(INP_FILE)
    val parsed = file.map(parseLine)

    // Split the data, 60% training data, and 40% testing data
    val splits = parsed.randomSplit(Array(0.6,0.4))
    val training = splits(0).cache()
    val testing = splits(1).cache()

//    val impurity = "variance"

//    val Algo   = "Classification"
    val maxDepth = 5
    val numClasses = 7


    val model = DecisionTree.train(training, Classification, Entropy, maxDepth, numClasses)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testing.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }.mean()
    val testErr = labelsAndPredictions.filter(r => r._1 != r._2).count().toDouble / testing.count()
    println("Decision Tree Accuracy = " + testErr)
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression tree model:\n" + model.toDebugString)

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(labelsAndPredictions)
    val precision = metrics.precision
    println("Precision ="+ precision)


    // Calculating for Naive Bayes
    val model2 = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel2 = testing.map(p => (model2.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel2.filter(x => x._1 == x._2).count() / testing.count()
    println("Naive Bayes Accuracy = " + accuracy)

  }

}
