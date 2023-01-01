name := "dip22-assignment"
version := "1.0"
scalaVersion := "2.12.16"

val SparkVersion: String = "3.3.0"
val ScalaTestVersion: String = "3.2.13"

libraryDependencies += "org.apache.spark" %% "spark-core" % SparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % SparkVersion
libraryDependencies += "org.apache.spark" %% "spark-sql" % SparkVersion
libraryDependencies += "org.apache.spark" %% "spark-streaming" % SparkVersion
libraryDependencies += "org.scalatest" %% "scalatest" % ScalaTestVersion % "test"

// suppress all log messages when setting up the Spark Session
javaOptions += "-Dlog4j.configurationFile=project/log4j.properties"

// in order to be able to use the javaOptions defined above
Test / fork := true
