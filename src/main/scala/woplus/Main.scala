package woplus

import ch04.DTreeUtil
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import scala.collection.immutable.IndexedSeq
import scala.collection.mutable.ArrayBuffer
import tw.com.chttl.spark.mllib.util.NAStat
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}

/**
 * Created by leorick on 2016/4/21.
 */
object Main {
  val appName = "woplus.device"








  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)
  }
}
