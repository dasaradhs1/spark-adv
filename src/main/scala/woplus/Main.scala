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






  def profile(sc:SparkContext) = {
    /*
wc -l /media/sf_WORKSPACE.2W/dataset/woplus/userprofile/profile.csv
2391167 /media/sf_WORKSPACE.2W/dataset/woplus/userprofile/profile.csv
     */
    val srcProfile = sc.textFile("file:///home/leo/woplus/userprofile/").
      map{_.split(",")}.
      filter{ case toks => !toks(0).contains("IMEI") }.
      cache
    /*
    NAStat.statsWithMissing( srcProfile.map{ toks => Array(toks.size.toDouble)} )
Array(stats: (count: 2391166, mean: 24.000000, stdev: 0.000000, max: 24.000000, min: 24.000000), NaN: 0)
     */
    srcProfile.unpersist(true)
  }


  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)
  }
}
