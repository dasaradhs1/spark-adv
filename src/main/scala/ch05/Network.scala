package ch05

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Matrix, Vectors}
import org.apache.spark.mllib.regression.{RidgeRegressionModel, RidgeRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import tw.com.chttl.spark.mllib.util.{DTUtil, LRUtil, StatsUtil}
import tw.com.chttl.spark.test.util.TimeEvaluation

import scala.collection.immutable.IndexedSeq

/**
 * Created by leorick on 2016/2/22.
 */
object Network {
  val appName = ""

  def loadData(sc:SparkContext): RDD[(String, Vector)] = {
    sc.textFile("file:///home/leoricklin/dataset/kdd1999/kddcup.data").
      map { line =>
      val buffer = line.split(',').toBuffer
      buffer.remove(1, 3)
      val label = buffer.remove(buffer.length-1)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label,vector)
    }
  }

  def correlation(vecs:RDD[Vector], names:Seq[String]): String = {
    vecs.cache()
    val res: Matrix = Statistics.corr(vecs, "pearson")
    vecs.unpersist(true)
    StatsUtil.printMatrix(res, Some(names), 10)
  }

  def L2(lps: RDD[LabeledPoint]): RidgeRegressionModel = {
    lps.cache()
    val iter = 100
    val modelRidge = RidgeRegressionWithSGD.train(lps, iter)
    lps.unpersist(true)
    modelRidge
  }

  def selectFeatures( lps: RDD[LabeledPoint] , numClasses: Int
    , numIteration:Int = 3 , minFeatureOccurRatio:Double = 0.2
    , numTrees:Int = 10, featureSubset:String = "auto", impurity:String = "entropy", depth:Int = 4, bins:Int = 100
    , seed:Int = 100 , sorted:Boolean = true): Seq[(Int, (Int, Double))] = {
    lps.cache()
    // 每個feature在tree出現最少次數
    val minFeatureOccurs = BigDecimal(((java.lang.Math.pow(2, depth+1) -1) * minFeatureOccurRatio)).
      setScale(0, scala.math.BigDecimal.RoundingMode.UP).toInt // 7
    val featureSelects: Map[Int, (Int, Double)] = (1 to numIteration).
        map{ i =>
          val modelRf = RandomForest.trainClassifier(lps, numClasses, Map[Int,Int]()
            , numTrees, featureSubset, impurity, depth, bins, seed)
          // this metric cloud be used for model selection
          /*
          val precision = (new MulticlassMetrics( lps.
            zip( modelRf.predict( lps.map{ lp => lp.features }) ).
            map{ case (lp: LabeledPoint, predict: Double) => (lp.label, predict) } ) ).precision
           */
          val nodeAvgGain: Map[Int, (Int, Double)] = DTUtil.nodeTreeAvgGain(modelRf.trees)
          nodeAvgGain.filter{ case (k, (cnt, gain)) => cnt >= minFeatureOccurs }  }.
        reduce{ (map1, map2) =>
          map1 ++  map2.
            map{ case (id2:Int, (cnt2:Int, avg2:Double)) =>
              val (cnt1:Int, avg1:Double) = map1.getOrElse(id2, 0 -> 0.0)
              val cnt = cnt1+cnt2
              id2 -> (cnt, ((cnt1*avg1+cnt2*avg2).toDouble)/cnt ) } }
    lps.unpersist(true)
    if (sorted) {
      featureSelects.toSeq.sortBy{ case (k, (cnt, gain)) => gain }.reverse
    } else {
      featureSelects.toSeq
    }
  }

  def fitRfModel(lps: RDD[LabeledPoint]): Array[(RandomForestModel, Array[Double])] = {
    val numFolds = 4
    val numTrees = 5
    val depth = 4
    val bins = 32
    val numClasses = 23
    val catInfo = Map[Int,Int]()
    val impurity = "entropy"
    val seed = 100
    val folds: Array[(RDD[LabeledPoint], RDD[LabeledPoint])] = MLUtils.kFold(lps, numFolds, seed)
    val modelMetrics: Array[(RandomForestModel, Array[Double])] = folds.
      map{ case (training, validation) =>
      training.cache;validation.cache()
      // training model
      val model = RandomForest.trainClassifier(training, numClasses, catInfo, numTrees, "auto", impurity, depth, bins, seed)
      training.unpersist()
      // validatiing model
      val predictLabels = model.predict( validation.map(_.features) ).
        zip( validation.map(_.label) )
      validation.unpersist()
      val metric = new MulticlassMetrics(predictLabels)
      ( model, Array(metric.fMeasure, metric.precision, metric.recall) ) }
    modelMetrics
  }

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)
    val labelsAndData: RDD[(String, Vector)] = loadData(sc)
    val labels: Seq[(String, Long)] = labelsAndData.keys.countByValue().toSeq.sortBy(_._2).reverse
    /*
    labels.foreach(println)
(smurf.,2807886)
(neptune.,1072017)
(normal.,972781)
(satan.,15892)
(ipsweep.,12481)
(portsweep.,10413)
(nmap.,2316)
(back.,2203)
(warezclient.,1020)
(teardrop.,979)
(pod.,264)
(guess_passwd.,53)
(buffer_overflow.,30)
(land.,21)
(warezmaster.,20)
(imap.,12)
(rootkit.,10)
(loadmodule.,9)
(ftp_write.,8)
(multihop.,7)
(phf.,4)
(perl.,3)
(spy.,2)
     */
    val names = Map(0->"duration", 1->"src_bytes", 2->"dst_bytes", 3->"land", 4->"wrong_fragment"
      ,5->"urgent", 6->"hot", 7->"num_failed_logins", 8->"logged_in", 9->"num_compromised"
      ,10->"root_shell", 11->"su_attempted", 12->"num_root", 13->"num_file_creations", 14->"num_shells"
      ,15->"num_access_files", 16->"num_outbound_cmds", 17->"is_host_login", 18->"is_guest_login", 19->"count"
      ,20->"srv_count", 21->"serror_rate", 22->"srv_serror_rate", 23->"rerror_rate", 24->"srv_rerror_rate"
      ,25->"same_srv_rate", 26->"diff_srv_rate", 27->"srv_diff_host_rate", 28->"dst_host_count", 29->"dst_host_srv_count"
      ,30->"dst_host_same_srv_rate", 31->"dst_host_diff_srv_rate", 32->"dst_host_same_src_port_rate", 33->"dst_host_srv_diff_host_rate", 34->"dst_host_serror_rate"
      ,35->"dst_host_srv_serror_rate", 36->"dst_host_rerror_rate", 37->"dst_host_srv_rerror_rate")
    // evaluate correlation
    val retCorr = correlation(labelsAndData.values, names.toSeq.sortBy{ case (idx, name) => idx}.map{case (idx, name) => name})
    /*
    print(retCorr)
            |duration    |src_bytes   |dst_bytes   |land        |wrong_fragme|urgent      |hot         |num_failed_l|logged_in   |num_compromi|root_shell  |su_attempted|num_root    |num_file_cre|num_shells  |num_access_f|num_outbound|is_host_logi|is_guest_log|count       |srv_count   |serror_rate |srv_serror_r|rerror_rate |srv_rerror_r|same_srv_rat|diff_srv_rat|srv_diff_hos|dst_host_cou|dst_host_srv|dst_host_sam|dst_host_dif|dst_host_sam|dst_host_srv|dst_host_ser|dst_host_srv|dst_host_rer|dst_host_srv
duration    |+1.000      |+0.041      |+0.020      |-0.000      |-0.001      |+0.004      |+0.004      |+0.007      |-0.021      |+0.027      |+0.026      |+0.052      |+0.029      |+0.095      |-0.000      |+0.024      |NaN         |-0.000      |+0.002      |-0.105      |-0.080      |-0.031      |-0.031      |+0.017      |+0.017      |+0.022      |+0.050      |-0.013      |+0.011      |-0.117      |-0.119      |+0.409      |+0.043      |-0.009      |-0.031      |-0.031      |+0.011      |+0.016
src_bytes   |+0.041      |+1.000      |+0.000      |-0.000      |-0.000      |-0.000      |+0.001      |-0.000      |+0.000      |+0.000      |-0.000      |-0.000      |-0.000      |+0.000      |+0.000      |-0.000      |NaN         |-0.000      |-0.000      |-0.002      |-0.001      |-0.001      |-0.001      |+0.003      |+0.003      |+0.001      |+0.000      |-0.000      |-0.002      |-0.002      |-0.002      |+0.001      |-0.001      |+0.000      |-0.001      |-0.001      |-0.000      |+0.003
dst_bytes   |+0.020      |+0.000      |+1.000      |-0.000      |-0.000      |+0.000      |+0.000      |+0.001      |+0.002      |+0.001      |+0.001      |+0.001      |+0.001      |+0.000      |-0.000      |+0.000      |NaN         |+0.000      |+0.000      |-0.003      |-0.002      |-0.001      |-0.001      |+0.002      |+0.002      |+0.001      |-0.000      |+0.000      |-0.002      |-0.001      |-0.001      |+0.003      |-0.001      |+0.000      |-0.001      |-0.001      |+0.003      |+0.003
land        |-0.000      |-0.000      |-0.000      |+1.000      |-0.000      |-0.000      |-0.000      |-0.000      |-0.001      |-0.000      |-0.000      |-0.000      |-0.000      |-0.000      |-0.000      |-0.000      |NaN         |-0.000      |-0.000      |-0.004      |-0.003      |+0.005      |+0.005      |-0.000      |-0.001      |+0.001      |+0.001      |+0.013      |-0.009      |-0.004      |+0.001      |-0.000      |+0.001      |+0.033      |+0.005      |+0.003      |-0.001      |-0.001
wrong_fragme|-0.001      |-0.000      |-0.000      |-0.000      |+1.000      |-0.000      |-0.000      |-0.000      |-0.006      |-0.000      |-0.000      |-0.000      |-0.000      |-0.000      |-0.000      |-0.000      |NaN         |-0.000      |-0.000      |-0.020      |-0.015      |-0.004      |-0.007      |-0.004      |-0.004      |+0.006      |-0.002      |+0.000      |-0.002      |-0.019      |-0.017      |+0.023      |-0.010      |+0.004      |-0.006      |-0.007      |+0.009      |-0.004
urgent      |+0.004      |-0.000      |+0.000      |-0.000      |-0.000      |+1.000      |+0.004      |+0.031      |+0.003      |+0.018      |+0.089      |+0.133      |+0.031      |+0.012      |+0.003      |+0.024      |NaN         |-0.000      |-0.000      |-0.002      |-0.001      |-0.001      |-0.001      |-0.000      |-0.000      |+0.001      |-0.000      |-0.000      |-0.003      |-0.002      |-0.002      |+0.002      |-0.001      |+0.003      |-0.000      |-0.001      |-0.000      |-0.000
hot         |+0.004      |+0.001      |+0.000      |-0.000      |-0.000      |+0.004      |+1.000      |+0.004      |+0.065      |+0.003      |+0.018      |+0.002      |+0.002      |+0.020      |+0.002      |+0.001      |NaN         |+0.001      |+0.804      |-0.042      |-0.032      |-0.012      |-0.012      |-0.006      |-0.005      |+0.013      |+0.004      |-0.000      |-0.036      |-0.033      |-0.025      |+0.009      |-0.032      |-0.003      |-0.012      |-0.012      |-0.005      |-0.005
num_failed_l|+0.007      |-0.000      |+0.001      |-0.000      |-0.000      |+0.031      |+0.004      |+1.000      |+0.002      |+0.020      |+0.024      |+0.069      |+0.019      |+0.014      |-0.000      |+0.001      |NaN         |-0.000      |+0.005      |-0.007      |-0.005      |-0.002      |-0.002      |+0.006      |+0.006      |+0.002      |+0.001      |-0.001      |-0.010      |-0.007      |-0.003      |+0.002      |-0.005      |+0.004      |-0.001      |-0.001      |+0.005      |+0.005
logged_in   |-0.021      |+0.000      |+0.002      |-0.001      |-0.006      |+0.003      |+0.065      |+0.002      |+1.000      |+0.005      |+0.020      |+0.011      |+0.008      |+0.023      |+0.021      |+0.070      |NaN         |+0.002      |+0.071      |-0.631      |-0.473      |-0.189      |-0.189      |-0.099      |-0.097      |+0.217      |-0.071      |+0.338      |-0.628      |+0.126      |+0.157      |-0.059      |-0.461      |+0.140      |-0.188      |-0.189      |-0.091      |-0.088
num_compromi|+0.027      |+0.000      |+0.001      |-0.000      |-0.000      |+0.018      |+0.003      |+0.020      |+0.005      |+1.000      |+0.172      |+0.350      |+0.998      |+0.012      |+0.001      |+0.144      |NaN         |+0.001      |-0.000      |-0.003      |-0.003      |-0.001      |-0.001      |-0.000      |-0.000      |+0.001      |-0.000      |-0.000      |-0.005      |-0.003      |-0.002      |+0.002      |-0.002      |+0.003      |-0.000      |-0.001      |-0.000      |-0.000
root_shell  |+0.026      |-0.000      |+0.001      |-0.000      |-0.000      |+0.089      |+0.018      |+0.024      |+0.020      |+0.172      |+1.000      |+0.456      |+0.187      |+0.035      |+0.037      |+0.132      |NaN         |-0.000      |-0.000      |-0.013      |-0.010      |-0.003      |-0.003      |-0.001      |-0.001      |+0.004      |-0.001      |+0.001      |-0.011      |-0.003      |-0.000      |+0.001      |-0.009      |+0.009      |-0.003      |-0.003      |-0.001      |-0.001
su_attempted|+0.052      |-0.000      |+0.001      |-0.000      |-0.000      |+0.133      |+0.002      |+0.069      |+0.011      |+0.350      |+0.456      |+1.000      |+0.378      |+0.028      |+0.003      |+0.259      |NaN         |-0.000      |-0.000      |-0.007      |-0.005      |-0.001      |-0.001      |-0.000      |-0.001      |+0.002      |-0.001      |-0.000      |-0.008      |-0.007      |-0.005      |+0.003      |-0.005      |+0.006      |-0.000      |-0.000      |-0.000      |-0.000
num_root    |+0.029      |-0.000      |+0.001      |-0.000      |-0.000      |+0.031      |+0.002      |+0.019      |+0.008      |+0.998      |+0.187      |+0.378      |+1.000      |+0.011      |+0.001      |+0.157      |NaN         |+0.002      |-0.000      |-0.005      |-0.004      |-0.001      |-0.001      |-0.001      |-0.001      |+0.002      |-0.000      |+0.000      |-0.007      |-0.005      |-0.004      |+0.003      |-0.003      |+0.004      |-0.001      |-0.001      |-0.001      |-0.001
num_file_cre|+0.095      |+0.000      |+0.000      |-0.000      |-0.000      |+0.012      |+0.020      |+0.014      |+0.023      |+0.012      |+0.035      |+0.028      |+0.011      |+1.000      |+0.010      |+0.078      |NaN         |-0.000      |+0.001      |-0.015      |-0.011      |-0.004      |-0.004      |-0.002      |-0.002      |+0.005      |-0.000      |+0.006      |-0.017      |-0.012      |-0.009      |+0.006      |-0.010      |+0.006      |-0.002      |-0.002      |-0.001      |-0.001
num_shells  |-0.000      |+0.000      |-0.000      |-0.000      |-0.000      |+0.003      |+0.002      |-0.000      |+0.021      |+0.001      |+0.037      |+0.003      |+0.001      |+0.010      |+1.000      |+0.003      |NaN         |-0.000      |-0.000      |-0.013      |-0.010      |-0.004      |-0.004      |-0.002      |-0.002      |+0.004      |+0.000      |-0.000      |-0.014      |-0.010      |-0.008      |+0.002      |-0.004      |+0.003      |-0.004      |-0.004      |-0.002      |-0.002
num_access_f|+0.024      |-0.000      |+0.000      |-0.000      |-0.000      |+0.024      |+0.001      |+0.001      |+0.070      |+0.144      |+0.132      |+0.259      |+0.157      |+0.078      |+0.003      |+1.000      |NaN         |-0.000      |-0.000      |-0.045      |-0.034      |-0.013      |-0.013      |-0.007      |-0.005      |+0.015      |-0.003      |+0.029      |-0.019      |-0.000      |+0.005      |+0.002      |-0.034      |+0.002      |-0.013      |-0.013      |-0.007      |-0.007
num_outbound|NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |+1.000      |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN         |NaN
is_host_logi|-0.000      |-0.000      |+0.000      |-0.000      |-0.000      |-0.000      |+0.001      |-0.000      |+0.002      |+0.001      |-0.000      |-0.000      |+0.002      |-0.000      |-0.000      |-0.000      |NaN         |+1.000      |-0.000      |-0.001      |-0.001      |+0.000      |+0.001      |-0.000      |-0.000      |-0.000      |+0.004      |-0.000      |+0.000      |-0.001      |-0.001      |+0.002      |-0.001      |-0.000      |+0.000      |+0.001      |-0.000      |-0.000
is_guest_log|+0.002      |-0.000      |+0.000      |-0.000      |-0.000      |-0.000      |+0.804      |+0.005      |+0.071      |-0.000      |-0.000      |-0.000      |-0.000      |+0.001      |-0.000      |-0.000      |NaN         |-0.000      |+1.000      |-0.046      |-0.035      |-0.013      |-0.013      |-0.007      |-0.007      |+0.014      |+0.007      |-0.003      |-0.044      |-0.043      |-0.034      |+0.013      |-0.035      |-0.004      |-0.013      |-0.013      |-0.006      |-0.006
count       |-0.105      |-0.002      |-0.003      |-0.004      |-0.020      |-0.002      |-0.042      |-0.007      |-0.631      |-0.003      |-0.013      |-0.007      |-0.005      |-0.015      |-0.013      |-0.045      |NaN         |-0.001      |-0.046      |+1.000      |+0.943      |-0.319      |-0.319      |-0.211      |-0.211      |+0.364      |-0.180      |-0.314      |+0.534      |+0.515      |+0.476      |-0.266      |+0.863      |-0.245      |-0.320      |-0.319      |-0.214      |-0.212
srv_count   |-0.080      |-0.001      |-0.002      |-0.003      |-0.015      |-0.001      |-0.032      |-0.005      |-0.473      |-0.003      |-0.010      |-0.005      |-0.004      |-0.011      |-0.010      |-0.034      |NaN         |-0.001      |-0.035      |+0.943      |+1.000      |-0.538      |-0.538      |-0.290      |-0.290      |+0.625      |-0.300      |-0.235      |+0.401      |+0.721      |+0.695      |-0.332      |+0.947      |-0.183      |-0.539      |-0.538      |-0.293      |-0.291
serror_rate |-0.031      |-0.001      |-0.001      |+0.005      |-0.004      |-0.001      |-0.012      |-0.002      |-0.189      |-0.001      |-0.003      |-0.001      |-0.001      |-0.004      |-0.004      |-0.013      |NaN         |+0.000      |-0.013      |-0.319      |-0.538      |+1.000      |+0.999      |-0.112      |-0.112      |-0.858      |+0.251      |-0.092      |+0.156      |-0.782      |-0.803      |+0.161      |-0.585      |-0.072      |+0.999      |+0.998      |-0.113      |-0.113
srv_serror_r|-0.031      |-0.001      |-0.001      |+0.005      |-0.007      |-0.001      |-0.012      |-0.002      |-0.189      |-0.001      |-0.003      |-0.001      |-0.001      |-0.004      |-0.004      |-0.013      |NaN         |+0.001      |-0.013      |-0.319      |-0.538      |+0.999      |+1.000      |-0.112      |-0.115      |-0.857      |+0.252      |-0.092      |+0.156      |-0.781      |-0.802      |+0.161      |-0.584      |-0.072      |+0.998      |+0.999      |-0.113      |-0.116
rerror_rate |+0.017      |+0.003      |+0.002      |-0.000      |-0.004      |-0.000      |-0.006      |+0.006      |-0.099      |-0.000      |-0.001      |-0.000      |-0.001      |-0.002      |-0.002      |-0.007      |NaN         |-0.000      |-0.007      |-0.211      |-0.290      |-0.112      |-0.112      |+1.000      |+0.995      |-0.330      |+0.234      |+0.023      |-0.091      |-0.333      |-0.322      |+0.214      |-0.267      |+0.143      |-0.112      |-0.112      |+0.990      |+0.986
srv_rerror_r|+0.017      |+0.003      |+0.002      |-0.001      |-0.004      |-0.000      |-0.005      |+0.006      |-0.097      |-0.000      |-0.001      |-0.001      |-0.001      |-0.002      |-0.002      |-0.005      |NaN         |-0.000      |-0.007      |-0.211      |-0.290      |-0.112      |-0.115      |+0.995      |+1.000      |-0.329      |+0.233      |+0.025      |-0.089      |-0.332      |-0.321      |+0.213      |-0.267      |+0.140      |-0.112      |-0.115      |+0.986      |+0.988
same_srv_rat|+0.022      |+0.001      |+0.001      |+0.001      |+0.006      |+0.001      |+0.013      |+0.002      |+0.217      |+0.001      |+0.004      |+0.002      |+0.002      |+0.005      |+0.004      |+0.015      |NaN         |-0.000      |+0.014      |+0.364      |+0.625      |-0.858      |-0.857      |-0.330      |-0.329      |+1.000      |-0.425      |+0.105      |-0.180      |+0.908      |+0.932      |-0.269      |+0.668      |+0.083      |-0.858      |-0.858      |-0.331      |-0.332
diff_srv_rat|+0.050      |+0.000      |-0.000      |+0.001      |-0.002      |-0.000      |+0.004      |+0.001      |-0.071      |-0.000      |-0.001      |-0.001      |-0.000      |-0.000      |+0.000      |-0.003      |NaN         |+0.004      |+0.007      |-0.180      |-0.300      |+0.251      |+0.252      |+0.234      |+0.233      |-0.425      |+1.000      |-0.022      |+0.054      |-0.418      |-0.426      |+0.533      |-0.268      |-0.027      |+0.251      |+0.252      |+0.228      |+0.234
srv_diff_hos|-0.013      |-0.000      |+0.000      |+0.013      |+0.000      |-0.000      |-0.000      |-0.001      |+0.338      |-0.000      |+0.001      |-0.000      |+0.000      |+0.006      |-0.000      |+0.029      |NaN         |-0.000      |-0.003      |-0.314      |-0.235      |-0.092      |-0.092      |+0.023      |+0.025      |+0.105      |-0.022      |+1.000      |-0.384      |+0.007      |+0.047      |-0.007      |-0.189      |+0.256      |-0.091      |-0.093      |+0.024      |+0.020
dst_host_cou|+0.011      |-0.002      |-0.002      |-0.009      |-0.002      |-0.003      |-0.036      |-0.010      |-0.628      |-0.005      |-0.011      |-0.008      |-0.007      |-0.017      |-0.014      |-0.019      |NaN         |+0.000      |-0.044      |+0.534      |+0.401      |+0.156      |+0.156      |-0.091      |-0.089      |-0.180      |+0.054      |-0.384      |+1.000      |-0.041      |-0.124      |+0.025      |+0.298      |-0.487      |+0.156      |+0.157      |-0.095      |-0.090
dst_host_srv|-0.117      |-0.002      |-0.001      |-0.004      |-0.019      |-0.002      |-0.033      |-0.007      |+0.126      |-0.003      |-0.003      |-0.007      |-0.005      |-0.012      |-0.010      |-0.000      |NaN         |-0.001      |-0.043      |+0.515      |+0.721      |-0.782      |-0.781      |-0.333      |-0.332      |+0.908      |-0.418      |+0.007      |-0.041      |+1.000      |+0.979      |-0.467      |+0.684      |-0.008      |-0.783      |-0.782      |-0.334      |-0.334
dst_host_sam|-0.119      |-0.002      |-0.001      |+0.001      |-0.017      |-0.002      |-0.025      |-0.003      |+0.157      |-0.002      |-0.000      |-0.005      |-0.004      |-0.009      |-0.008      |+0.005      |NaN         |-0.001      |-0.034      |+0.476      |+0.695      |-0.803      |-0.802      |-0.322      |-0.321      |+0.932      |-0.426      |+0.047      |-0.124      |+0.979      |+1.000      |-0.472      |+0.676      |+0.054      |-0.804      |-0.803      |-0.323      |-0.324
dst_host_dif|+0.409      |+0.001      |+0.003      |-0.000      |+0.023      |+0.002      |+0.009      |+0.002      |-0.059      |+0.002      |+0.001      |+0.003      |+0.003      |+0.006      |+0.002      |+0.002      |NaN         |+0.002      |+0.013      |-0.266      |-0.332      |+0.161      |+0.161      |+0.214      |+0.213      |-0.269      |+0.533      |-0.007      |+0.025      |-0.467      |-0.472      |+1.000      |-0.159      |+0.005      |+0.161      |+0.161      |+0.218      |+0.215
dst_host_sam|+0.043      |-0.001      |-0.001      |+0.001      |-0.010      |-0.001      |-0.032      |-0.005      |-0.461      |-0.002      |-0.009      |-0.005      |-0.003      |-0.010      |-0.004      |-0.034      |NaN         |-0.001      |-0.035      |+0.863      |+0.947      |-0.585      |-0.584      |-0.267      |-0.267      |+0.668      |-0.268      |-0.189      |+0.298      |+0.684      |+0.676      |-0.159      |+1.000      |-0.074      |-0.585      |-0.584      |-0.269      |-0.270
dst_host_srv|-0.009      |+0.000      |+0.000      |+0.033      |+0.004      |+0.003      |-0.003      |+0.004      |+0.140      |+0.003      |+0.009      |+0.006      |+0.004      |+0.006      |+0.003      |+0.002      |NaN         |-0.000      |-0.004      |-0.245      |-0.183      |-0.072      |-0.072      |+0.143      |+0.140      |+0.083      |-0.027      |+0.256      |-0.487      |-0.008      |+0.054      |+0.005      |-0.074      |+1.000      |-0.072      |-0.072      |+0.145      |+0.146
dst_host_ser|-0.031      |-0.001      |-0.001      |+0.005      |-0.006      |-0.000      |-0.012      |-0.001      |-0.188      |-0.000      |-0.003      |-0.000      |-0.001      |-0.002      |-0.004      |-0.013      |NaN         |+0.000      |-0.013      |-0.320      |-0.539      |+0.999      |+0.998      |-0.112      |-0.112      |-0.858      |+0.251      |-0.091      |+0.156      |-0.783      |-0.804      |+0.161      |-0.585      |-0.072      |+1.000      |+0.998      |-0.114      |-0.113
dst_host_srv|-0.031      |-0.001      |-0.001      |+0.003      |-0.007      |-0.001      |-0.012      |-0.001      |-0.189      |-0.001      |-0.003      |-0.000      |-0.001      |-0.002      |-0.004      |-0.013      |NaN         |+0.001      |-0.013      |-0.319      |-0.538      |+0.998      |+0.999      |-0.112      |-0.115      |-0.858      |+0.252      |-0.093      |+0.157      |-0.782      |-0.803      |+0.161      |-0.584      |-0.072      |+0.998      |+1.000      |-0.113      |-0.116
dst_host_rer|+0.011      |-0.000      |+0.003      |-0.001      |+0.009      |-0.000      |-0.005      |+0.005      |-0.091      |-0.000      |-0.001      |-0.000      |-0.001      |-0.001      |-0.002      |-0.007      |NaN         |-0.000      |-0.006      |-0.214      |-0.293      |-0.113      |-0.113      |+0.990      |+0.986      |-0.331      |+0.228      |+0.024      |-0.095      |-0.334      |-0.323      |+0.218      |-0.269      |+0.145      |-0.114      |-0.113      |+1.000      |+0.987
dst_host_srv|+0.016      |+0.003      |+0.003      |-0.001      |-0.004      |-0.000      |-0.005      |+0.005      |-0.088      |-0.000      |-0.001      |-0.000      |-0.001      |-0.001      |-0.002      |-0.007      |NaN         |-0.000      |-0.006      |-0.212      |-0.291      |-0.113      |-0.116      |+0.986      |+0.988      |-0.332      |+0.234      |+0.020      |-0.090      |-0.334      |-0.324      |+0.215      |-0.270      |+0.146      |-0.113      |-0.116      |+0.987      |+1.000
     */
    // RF baseline
    val bvLabelsIdx = sc.broadcast( labels.map{ case (label, cnt) => label}.zipWithIndex.toMap )
    val numClasses = labels.size // 23
    val lps = labelsAndData.mapPartitions{ ite =>
      val labelIdx: Map[String, Int] = bvLabelsIdx.value
      ite.map{ case (label: String, vecs: Vector) =>
        new LabeledPoint( labelIdx.getOrElse(label, -1).toDouble, vecs) }
    }
    val modelRfs: Array[(RandomForestModel, Array[Double])] = fitRfModel(lps)
    /*
Array(
(TreeEnsembleModel classifier with 5 trees, Array(0.9924345984501944, 0.9924345984501944, 0.9924345984501944)),
(TreeEnsembleModel classifier with 5 trees, Array(0.993263100412585,  0.993263100412585,  0.993263100412585)),
(TreeEnsembleModel classifier with 5 trees, Array(0.9897629138845836, 0.9897629138845836, 0.9897629138845836)),
(TreeEnsembleModel classifier with 5 trees, Array(0.9904238274966071, 0.9904238274966071, 0.9904238274966071)))
     */
    modelRfs.map{ case (model, metrics) => metrics }.
      reduce( (a1, a2) => Array(a1(0)+a2(0), a1(1)+a2(1), a1(2)+a2(2)) ).
      map{ measure => measure / 4}
    /*
Array(0.9914711100609925, 0.9914711100609925, 0.9914711100609925)
     */
    // feature selection
    val multiSelect: Seq[(Int, (Int, Double))] = selectFeatures(lps, numClasses)
    multiSelect.foreach{ case (k, (cnt, avg)) => println(f"feature=${k}%-2s(${names.getOrElse(k, "NA")}%-20.20s), cnt=${cnt}%6d, avg.gain=${avg}%8.5f") }
    /* 1st round
feature=25(same_srv_rate       ), cnt=    35, avg.gain= 0.39446
feature=20(srv_count           ), cnt=     7, avg.gain= 0.39203
feature=19(count               ), cnt=    48, avg.gain= 0.37605
feature=26(diff_srv_rate       ), cnt=    28, avg.gain= 0.32750
feature=1 (src_bytes           ), cnt=    14, avg.gain= 0.31712
feature=29(dst_host_srv_count  ), cnt=     7, avg.gain= 0.29064
feature=30(dst_host_same_srv_ra), cnt=    15, avg.gain= 0.26962
feature=31(dst_host_diff_srv_ra), cnt=    22, avg.gain= 0.26079
feature=32(dst_host_same_src_po), cnt=    44, avg.gain= 0.25959
feature=21(serror_rate         ), cnt=     7, avg.gain= 0.15595
     */
    /* 2nd round
feature=25(same_srv_rate       ), cnt=    35, avg.gain= 0.39652
feature=29(dst_host_srv_count  ), cnt=     7, avg.gain= 0.24851
feature=1 (src_bytes           ), cnt=    16, avg.gain= 0.24289
feature=21(serror_rate         ), cnt=    16, avg.gain= 0.24220
feature=33(dst_host_srv_diff_ho), cnt=     7, avg.gain= 0.20519
feature=2 (dst_bytes           ), cnt=    20, avg.gain= 0.16082
feature=32(dst_host_same_src_po), cnt=    36, avg.gain= 0.26535
feature=31(dst_host_diff_srv_ra), cnt=    24, avg.gain= 0.16774
feature=26(diff_srv_rate       ), cnt=    32, avg.gain= 0.31656
feature=8 (logged_in           ), cnt=     7, avg.gain= 0.06382
     */
    /* 3rd round
feature=25(same_srv_rate       ), cnt=    38, avg.gain= 0.43246
feature=19(count               ), cnt=    43, avg.gain= 0.41857
feature=20(srv_count           ), cnt=     9, avg.gain= 0.40913
feature=1 (src_bytes           ), cnt=     7, avg.gain= 0.33355
feature=26(diff_srv_rate       ), cnt=    30, avg.gain= 0.32014
feature=32(dst_host_same_src_po), cnt=    45, avg.gain= 0.31889
feature=21(serror_rate         ), cnt=     7, avg.gain= 0.28256
feature=31(dst_host_diff_srv_ra), cnt=    15, avg.gain= 0.24278
feature=30(dst_host_same_srv_ra), cnt=     8, avg.gain= 0.24077
feature=29(dst_host_srv_count  ), cnt=     9, avg.gain= 0.22204
feature=2 (dst_bytes           ), cnt=    25, avg.gain= 0.08917
     */
    val numFeatures = names.size
    val minFeatureRatio = 0.25
    val maxFeatureSelect = BigDecimal(numFeatures * minFeatureRatio ).
      setScale(0, scala.math.BigDecimal.RoundingMode.UP).toInt // 10
    val selectFeatureIds: Seq[Int] = multiSelect.take(maxFeatureSelect).
        map{ case (id, (cnt, gain)) => id } // selectFeatureIds.size = 10
    val bvSelectFeatures = sc.broadcast( selectFeatureIds )
    val lpSelects: RDD[LabeledPoint] = lps.mapPartitions{ (ite: Iterator[LabeledPoint]) =>
      val selectFeatures: Seq[Int] = bvSelectFeatures.value
      ite.map{ lp =>
        val featureSelect = selectFeatures.map{ id => lp.features(id) }.toArray
        new LabeledPoint( lp.label, Vectors.dense(featureSelect) ) } }
    // RF with feature selection
    val modelRfSelects: Array[(RandomForestModel, Array[Double])] = fitRfModel(lpSelects)
    /*
Array(
(TreeEnsembleModel classifier with 5 trees, Array(0.9935379881907918, 0.9935379881907918, 0.9935379881907918)),
(TreeEnsembleModel classifier with 5 trees, Array(0.9913158494943443, 0.9913158494943443, 0.9913158494943443)),
(TreeEnsembleModel classifier with 5 trees, Array(0.9928390182909461, 0.9928390182909461, 0.9928390182909461)),
(TreeEnsembleModel classifier with 5 trees, Array(0.99003757955646,   0.99003757955646,   0.99003757955646)))
     */
    modelRfSelects.map{ case (model, metrics) => metrics }.
      reduce( (a1, a2) => Array(a1(0)+a2(0), a1(1)+a2(1), a1(2)+a2(2)) ).
      map{ measure => measure / 4}
    /*
Array(0.9919326088831355, 0.9919326088831355, 0.9919326088831355)
     */

  }

}
