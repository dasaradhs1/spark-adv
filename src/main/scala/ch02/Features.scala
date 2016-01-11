package ch02

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by leorick on 2015/12/24.
 */
object Features {

  def main(args: Array[String]) {
    val appName = "MLlib Features"
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)
    // Stand Scaler
    val model1 = {
    }
  }
}
