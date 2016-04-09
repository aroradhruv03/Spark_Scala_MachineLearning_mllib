import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

import scala.collection.mutable.ArrayBuffer
//import scalax.io._
import java.io.File

/**
  * Created by dhruv on 4/7/16.
  */
object Q1 {

  val INP_FILE = "itemusermat"
  val INP_FILE_MOVIES = "movies.dat"
  val OUT_DIR = "/Users/dhruv/Documents/Bigdata/"

  def main (args: Array[String])
  {
    val conf = new SparkConf().setAppName("Q1.K-Means Clustering").setMaster("local")
    val sc = new SparkContext(conf)

    // Reading the itemusermat (Movie Rating) file
    val file = sc.textFile(INP_FILE)

    // Reading thr Movie Data file and then processing it
    val movieFile = sc.textFile(INP_FILE_MOVIES)
    val movieLine = movieFile.map(s => (s.split("::")) )
    val movieData = movieLine.map(s => (s(0).toInt, (s(1),s(2)) ) )

//    for(mov <- movieData.take(5)) println(mov) // Used to test if the file was read correctly

//    val line = sc.parallelize(file.take(10)) // Used for testing, only taking 10 Movie records
//    val line = sc.parallelize(file.take(10))

    // Taking the ratings and converting it to Vectors
    val parseData = file.map( s=> Vectors.dense( s.split(" ").drop(1).map(_.toDouble) ) ).cache()

    val numClusters = 10 // Value of K in Kmeans
    val clusters = KMeans.train(parseData, numClusters, 20)


    val NamesandData = file.map(s => (s.split(' ')(0).toInt, Vectors.dense(s.split(' ').drop(1).map(_.toDouble)))).cache()


    var unionData = (movieData.join (NamesandData)).cache()

    val groupedClusters = unionData.groupBy{rdd => clusters.predict(rdd._2._2)}
//
//
//
//    // Print out a list of the clusters and each point of the clusters
//    val groupedClusters = NamesandData.groupBy{rdd => clusters.predict(rdd._2)}

    val groupedCluster = groupedClusters.map(s => s )
//    val grouped = groupedClusters.map(s => (s._1, (s._2).map(d => d._1) )  )

    val grouped = groupedClusters.map(s => (s._1, s._2.map(d => (d._1,d._2._1)) )  )

//    val grpMov = grouped union(movieData)


    println("Grouped Clusters")
    groupedClusters.collect().foreach { println }

    println("Grouped Cluster")
    groupedCluster.collect().foreach { println }
    println("Grouped")
    grouped.collect().foreach { println }

    def myPrint(element : TraversableOnce[_]): Unit = {
    }

    var list = ArrayBuffer.empty[Any]

//    val file: Seekable =  Resource.fromFile(new File("scala-io.out"))

    for (grp <- grouped) {
      println("Cluster: " + grp._1)
      list += grp._1
      val movies = grp._2.take(5)
      for (movie <- movies)
        {
          println(movie)
          list += movie
        }
      println()
    }

    print("\nlis\n")
    list.foreach{ println }


    (sc.parallelize(list)).saveAsTextFile(OUT_DIR+"q1")

  }

}
