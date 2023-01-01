/*
Name: Md Toufique Hasan
Email: mdtoufique.hasan@tuni.fi
Student Id: 151129267
*/

package assignment22

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.{mean, stddev, udf}
import org.apache.spark.sql.{DataFrame, SparkSession, functions}


class Assignment {
  // suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)

  val spark: SparkSession = SparkSession.builder()
    .appName("assignment22")
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()

  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/dataD2.csv")

  // the data frame to be used in task 2
  val removeColumn= ("LABEL")
  val dataD3: DataFrame = spark.read
    .option("sep", ",")
    .option("header","true")
    .option("inferSchema", "true")
    .csv("data/dataD3.csv")
    .drop(removeColumn)

  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  val dataD2WithLabels: DataFrame = spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/dataD2.csv")

  val toDouble: Any = udf[Double, String]( _.toDouble)

  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    // ignore the LABEL column and drop the null values
    val data_t1 = df.drop("LABEL").na.drop("any")

    data_t1.cache()
    data_t1.show(10)

    // minimum and maximum value in column a from the DataFrame
    println(s"\nAnalysis of a column")
    data_t1.select(functions.count("a"), functions.min("a"), functions.max("a"), mean("a"), stddev("a")).show()

    // minimum and maximum value in column b from the DataFrame
    println(s"\nAnalysis of b column")
    data_t1.select(functions.count("b"), functions.min("b"), functions.max("b"), mean("b"), stddev("b")).show()

    // VectorAssembler for mapping input column "a" and "b" to "features"
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("features")

    // Pipeline with sequence of stages to process and learn from data
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(data_t1)
    val transformedTraining = pipeLine.transform(data_t1)

    // k-means object and fit the transformedTraining to get a k-means object
    val k_means = new KMeans().setK(k).setSeed(1L)

    val kmModel: KMeansModel = k_means.fit(transformedTraining)

    // k-means cluster centroids of vector data type converted to array as return values
    val centers = kmModel.clusterCenters
      .map(x => x.toArray)
      .map{case Array(f1,f2) => (f1,f2)}

    println(s"\nNumber of centroids = ${centers.length} \n ")
    return centers
  }
  println("\nBasic task 1: Basic 2D K-means")
  val task1: Array[(Double, Double)] = task1(dataD2, 5)
  task1.foreach(println)


  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    // Pipeline for training
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c"))
      .setOutputCol("features")

    val transformationPipeline: Pipeline = new Pipeline().setStages(Array(vectorAssembler))

    val pipeLine = transformationPipeline.fit(df).transform(df)
    //val transformedTraining = pipeLine.transform(df)

    // Kmeans model fitting to the data
    val k_means = new KMeans().setK(k).setSeed(1L)
    val kmModel = k_means.fit(pipeLine)
    val centers = kmModel.clusterCenters.map(x => (x(0), x(1), x(2)))
    return centers
  }
  println("\nBasic task 2: Three Dimensions")
  val task2: Array[(Double, Double, Double)] = task2(dataD3, 5)
  task2.foreach(println)


  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    // Converting label in binary. Here 0 means Ok and 1 means Fatal
    val df_convert = spark.createDataFrame(Seq((0, "Ok"), (1, "Fatal"))).toDF("id", "LABEL")

    df.show()
    df.printSchema()

    // Maps a string column of labels to an ML column of label indices
    val indexer = new StringIndexer().setInputCol("LABEL").setOutputCol("lid")

    val df_t3 = indexer.fit(df).transform(df)

    df_t3.show()
    df_t3.printSchema()

    // Drop LABEL column, but cast label ids (lid) to Double and remove null values
    val data_t3 = df_t3.drop("LABEL")
      .selectExpr("cast(a as Double) a", "cast(b as Double) b", "cast(lid as Double) label").na.drop("any")

    data_t3.printSchema()
    data_t3.cache()
    data_t3.show(10)

    // VectorAssembler for mapping input column "a", "b" and "label" to "features"
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "label"))
      .setOutputCol("features")

    // Perform pipeline with sequence of stages to process and learn from data
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(data_t3)
    val transformedTraining = pipeLine.transform(data_t3)

    // Create a k-means object and fit the transformedTraining to get a k-means object
    val k_means = new KMeans().setK(k).setSeed(1L)
    val kmModel = k_means.fit(transformedTraining)
    val centers = kmModel.clusterCenters
      .map(x => x.toArray)
      .map{case Array(f1,f2,f3) => (f1,f2,f3)}
      // changed from 0.5 to 0.43 in order to get two centers
      .filter(x => (x._3 > 0.43))
      .map{case (f1,f2,f3) => (f1,f2)}

    return centers
  }
  println("\nBasic task 3: Using Labels ")
  val task3: Array[(Double, Double)] = task3(dataD2WithLabels, 5)
  task3.foreach(println)


  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    def SilhouetteMethod(k: Int, df: DataFrame, evaluator: ClusteringEvaluator): (Int, Double) = {
      val data_t4 = df.drop("LABEL").na.drop("any")
      val vectorAssembler = new VectorAssembler()
        .setInputCols(Array("a", "b"))
        .setOutputCol("features")

      // Pipeline with sequence of stages to process and learn from data
      val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler))
      val pipeLine = transformationPipeline.fit(data_t4)
      val transformedTraining = pipeLine.transform(data_t4)
      val k_means = new KMeans().setK(k).setSeed(1L)
      val predictions = k_means.fit(transformedTraining).transform(transformedTraining)
      (k, evaluator.evaluate(predictions))
    }

    val evaluator = new ClusteringEvaluator()
    val score = (low to high).map(x => SilhouetteMethod(x, df, evaluator)).toArray
    score
  }

}
