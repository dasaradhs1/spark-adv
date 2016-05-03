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

  def correlation(vecs:RDD[Vector], fnames:Seq[String]): String = {
    vecs.cache()
    val res: Matrix = Statistics.corr(vecs, "pearson")
    vecs.unpersist(true)
    StatsUtil.corrMatrixTable(res, Some(fnames), 15) + "\n" +
    StatsUtil.sortCorrMatrix(res, Some(fnames)).map{ case (pair, coef) => f"[${pair}][${coef}%+-1.3f]" }.mkString("\n")
  }

  def L2(lps: RDD[LabeledPoint]): RidgeRegressionModel = {
    lps.cache()
    val iter = 100
    val modelRidge = RidgeRegressionWithSGD.train(lps, iter)
    lps.unpersist(true)
    modelRidge
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
    val retCorr = correlation(labelsAndData.values, names.toSeq.sortBy{ case (idx, name) => idx }.map{ case (idx, name) => name })
    /*
    print(retCorr)
               |duration       |src_bytes      |dst_bytes      |land           |
wrong_fragment |urgent         |hot            |num_failed_logi|logged_in      |
num_compromised|root_shell     |su_attempted   |num_root       |num_file_creati|num_shells     |num_access_file|num_outbound_cm|is_host_login  |is_guest_login |count          |srv_count      |serror_rate    |srv_serror_rate|rerror_rate    |srv_rerror_rate|same_srv_rate  |diff_srv_rate  |srv_diff_host_r|dst_host_count |dst_host_srv_co|dst_host_same_s|dst_host_diff_s|dst_host_same_s|dst_host_srv_di|dst_host_serror|dst_host_srv_se|dst_host_rerror|dst_host_srv_re
duration       |+1.000         |+0.041         |+0.020         |-0.000         |-0.001         |+0.004         |+0.004         |+0.007         |-0.021         |+0.027         |+0.026         |+0.052         |+0.029         |+0.095         |-0.000         |+0.024         |NaN            |-0.000         |+0.002         |-0.105         |-0.080         |-0.031         |-0.031         |+0.017         |+0.017         |+0.022         |+0.050         |-0.013         |+0.011         |-0.117         |-0.119         |+0.409         |+0.043         |-0.009         |-0.031         |-0.031         |+0.011         |+0.016
src_bytes      |+0.041         |+1.000         |+0.000         |-0.000         |-0.000         |-0.000         |+0.001         |-0.000         |+0.000         |+0.000         |-0.000         |-0.000         |-0.000         |+0.000         |+0.000         |-0.000         |NaN            |-0.000         |-0.000         |-0.002         |-0.001         |-0.001         |-0.001         |+0.003         |+0.003         |+0.001         |+0.000         |-0.000         |-0.002         |-0.002         |-0.002         |+0.001         |-0.001         |+0.000         |-0.001         |-0.001         |-0.000         |+0.003
dst_bytes      |+0.020         |+0.000         |+1.000         |-0.000         |-0.000         |+0.000         |+0.000         |+0.001         |+0.002         |+0.001         |+0.001         |+0.001         |+0.001         |+0.000         |-0.000         |+0.000         |NaN            |+0.000         |+0.000         |-0.003         |-0.002         |-0.001         |-0.001         |+0.002         |+0.002         |+0.001         |-0.000         |+0.000         |-0.002         |-0.001         |-0.001         |+0.003         |-0.001         |+0.000         |-0.001         |-0.001         |+0.003         |+0.003
land           |-0.000         |-0.000         |-0.000         |+1.000         |-0.000         |-0.000         |-0.000         |-0.000         |-0.001         |-0.000         |-0.000         |-0.000         |-0.000         |-0.000         |-0.000         |-0.000         |NaN            |-0.000         |-0.000         |-0.004         |-0.003         |+0.005         |+0.005         |-0.000         |-0.001         |+0.001         |+0.001         |+0.013         |-0.009         |-0.004         |+0.001         |-0.000         |+0.001         |+0.033         |+0.005         |+0.003         |-0.001         |-0.001
wrong_fragment |-0.001         |-0.000         |-0.000         |-0.000         |+1.000         |-0.000         |-0.000         |-0.000         |-0.006         |-0.000         |-0.000         |-0.000         |-0.000         |-0.000         |-0.000         |-0.000         |NaN            |-0.000         |-0.000         |-0.020         |-0.015         |-0.004         |-0.007         |-0.004         |-0.004         |+0.006         |-0.002         |+0.000         |-0.002         |-0.019         |-0.017         |+0.023         |-0.010         |+0.004         |-0.006         |-0.007         |+0.009         |-0.004
urgent         |+0.004         |-0.000         |+0.000         |-0.000         |-0.000         |+1.000         |+0.004         |+0.031         |+0.003         |+0.018         |+0.089         |+0.133         |+0.031         |+0.012         |+0.003         |+0.024         |NaN            |-0.000         |-0.000         |-0.002         |-0.001         |-0.001         |-0.001         |-0.000         |-0.000         |+0.001         |-0.000         |-0.000         |-0.003         |-0.002         |-0.002         |+0.002         |-0.001         |+0.003         |-0.000         |-0.001         |-0.000         |-0.000
hot            |+0.004         |+0.001         |+0.000         |-0.000         |-0.000         |+0.004         |+1.000         |+0.004         |+0.065         |+0.003         |+0.018         |+0.002         |+0.002         |+0.020         |+0.002         |+0.001         |NaN            |+0.001         |+0.804         |-0.042         |-0.032         |-0.012         |-0.012         |-0.006         |-0.005         |+0.013         |+0.004         |-0.000         |-0.036         |-0.033         |-0.025         |+0.009         |-0.032         |-0.003         |-0.012         |-0.012         |-0.005         |-0.005
num_failed_logi|+0.007         |-0.000         |+0.001         |-0.000         |-0.000         |+0.031         |+0.004         |+1.000         |+0.002         |+0.020         |+0.024         |+0.069         |+0.019         |+0.014         |-0.000         |+0.001         |NaN            |-0.000         |+0.005         |-0.007         |-0.005         |-0.002         |-0.002         |+0.006         |+0.006         |+0.002         |+0.001         |-0.001         |-0.010         |-0.007         |-0.003         |+0.002         |-0.005         |+0.004         |-0.001         |-0.001         |+0.005         |+0.005
logged_in      |-0.021         |+0.000         |+0.002         |-0.001         |-0.006         |+0.003         |+0.065         |+0.002         |+1.000         |+0.005         |+0.020         |+0.011         |+0.008         |+0.023         |+0.021         |+0.070         |NaN            |+0.002         |+0.071         |-0.631         |-0.473         |-0.189         |-0.189         |-0.099         |-0.097         |+0.217         |-0.071         |+0.338         |-0.628         |+0.126         |+0.157         |-0.059         |-0.461         |+0.140         |-0.188         |-0.189         |-0.091         |-0.088
num_compromised|+0.027         |+0.000         |+0.001         |-0.000         |-0.000         |+0.018         |+0.003         |+0.020         |+0.005         |+1.000         |+0.172         |+0.350         |+0.998         |+0.012         |+0.001         |+0.144         |NaN            |+0.001         |-0.000         |-0.003         |-0.003         |-0.001         |-0.001         |-0.000         |-0.000         |+0.001         |-0.000         |-0.000         |-0.005         |-0.003         |-0.002         |+0.002         |-0.002         |+0.003         |-0.000         |-0.001         |-0.000         |-0.000
root_shell     |+0.026         |-0.000         |+0.001         |-0.000         |-0.000         |+0.089         |+0.018         |+0.024         |+0.020         |+0.172         |+1.000         |+0.456         |+0.187         |+0.035         |+0.037         |+0.132         |NaN            |-0.000         |-0.000         |-0.013         |-0.010         |-0.003         |-0.003         |-0.001         |-0.001         |+0.004         |-0.001         |+0.001         |-0.011         |-0.003         |-0.000         |+0.001         |-0.009         |+0.009         |-0.003         |-0.003         |-0.001         |-0.001
su_attempted   |+0.052         |-0.000         |+0.001         |-0.000         |-0.000         |+0.133         |+0.002         |+0.069         |+0.011         |+0.350         |+0.456         |+1.000         |+0.378         |+0.028         |+0.003         |+0.259         |NaN            |-0.000         |-0.000         |-0.007         |-0.005         |-0.001         |-0.001         |-0.000         |-0.001         |+0.002         |-0.001         |-0.000         |-0.008         |-0.007         |-0.005         |+0.003         |-0.005         |+0.006         |-0.000         |-0.000         |-0.000         |-0.000
num_root       |+0.029         |-0.000         |+0.001         |-0.000         |-0.000         |+0.031         |+0.002         |+0.019         |+0.008         |+0.998         |+0.187         |+0.378         |+1.000         |+0.011         |+0.001         |+0.157         |NaN            |+0.002         |-0.000         |-0.005         |-0.004         |-0.001         |-0.001         |-0.001         |-0.001         |+0.002         |-0.000         |+0.000         |-0.007         |-0.005         |-0.004         |+0.003         |-0.003         |+0.004         |-0.001         |-0.001         |-0.001         |-0.001
num_file_creati|+0.095         |+0.000         |+0.000         |-0.000         |-0.000         |+0.012         |+0.020         |+0.014         |+0.023         |+0.012         |+0.035         |+0.028         |+0.011         |+1.000         |+0.010         |+0.078         |NaN            |-0.000         |+0.001         |-0.015         |-0.011         |-0.004         |-0.004         |-0.002         |-0.002         |+0.005         |-0.000         |+0.006         |-0.017         |-0.012         |-0.009         |+0.006         |-0.010         |+0.006         |-0.002         |-0.002         |-0.001         |-0.001
num_shells     |-0.000         |+0.000         |-0.000         |-0.000         |-0.000         |+0.003         |+0.002         |-0.000         |+0.021         |+0.001         |+0.037         |+0.003         |+0.001         |+0.010         |+1.000         |+0.003         |NaN            |-0.000         |-0.000         |-0.013         |-0.010         |-0.004         |-0.004         |-0.002         |-0.002         |+0.004         |+0.000         |-0.000         |-0.014         |-0.010         |-0.008         |+0.002         |-0.004         |+0.003         |-0.004         |-0.004         |-0.002         |-0.002
num_access_file|+0.024         |-0.000         |+0.000         |-0.000         |-0.000         |+0.024         |+0.001         |+0.001         |+0.070         |+0.144         |+0.132         |+0.259         |+0.157         |+0.078         |+0.003         |+1.000         |NaN            |-0.000         |-0.000         |-0.045         |-0.034         |-0.013         |-0.013         |-0.007         |-0.005         |+0.015         |-0.003         |+0.029         |-0.019         |-0.000         |+0.005         |+0.002         |-0.034         |+0.002         |-0.013         |-0.013         |-0.007         |-0.007
num_outbound_cm|NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |+1.000         |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN            |NaN
is_host_login  |-0.000         |-0.000         |+0.000         |-0.000         |-0.000         |-0.000         |+0.001         |-0.000         |+0.002         |+0.001         |-0.000         |-0.000         |+0.002         |-0.000         |-0.000         |-0.000         |NaN            |+1.000         |-0.000         |-0.001         |-0.001         |+0.000         |+0.001         |-0.000         |-0.000         |-0.000         |+0.004         |-0.000         |+0.000         |-0.001         |-0.001         |+0.002         |-0.001         |-0.000         |+0.000         |+0.001         |-0.000         |-0.000
is_guest_login |+0.002         |-0.000         |+0.000         |-0.000         |-0.000         |-0.000         |+0.804         |+0.005         |+0.071         |-0.000         |-0.000         |-0.000         |-0.000         |+0.001         |-0.000         |-0.000         |NaN            |-0.000         |+1.000         |-0.046         |-0.035         |-0.013         |-0.013         |-0.007         |-0.007         |+0.014         |+0.007         |-0.003         |-0.044         |-0.043         |-0.034         |+0.013         |-0.035         |-0.004         |-0.013         |-0.013         |-0.006         |-0.006
count          |-0.105         |-0.002         |-0.003         |-0.004         |-0.020         |-0.002         |-0.042         |-0.007         |-0.631         |-0.003         |-0.013         |-0.007         |-0.005         |-0.015         |-0.013         |-0.045         |NaN            |-0.001         |-0.046         |+1.000         |+0.943         |-0.319         |-0.319         |-0.211         |-0.211         |+0.364         |-0.180         |-0.314         |+0.534         |+0.515         |+0.476         |-0.266         |+0.863         |-0.245         |-0.320         |-0.319         |-0.214         |-0.212
srv_count      |-0.080         |-0.001         |-0.002         |-0.003         |-0.015         |-0.001         |-0.032         |-0.005         |-0.473         |-0.003         |-0.010         |-0.005         |-0.004         |-0.011         |-0.010         |-0.034         |NaN            |-0.001         |-0.035         |+0.943         |+1.000         |-0.538         |-0.538         |-0.290         |-0.290         |+0.625         |-0.300         |-0.235         |+0.401         |+0.721         |+0.695         |-0.332         |+0.947         |-0.183         |-0.539         |-0.538         |-0.293         |-0.291
serror_rate    |-0.031         |-0.001         |-0.001         |+0.005         |-0.004         |-0.001         |-0.012         |-0.002         |-0.189         |-0.001         |-0.003         |-0.001         |-0.001         |-0.004         |-0.004         |-0.013         |NaN            |+0.000         |-0.013         |-0.319         |-0.538         |+1.000         |+0.999         |-0.112         |-0.112         |-0.858         |+0.251         |-0.092         |+0.156         |-0.782         |-0.803         |+0.161         |-0.585         |-0.072         |+0.999         |+0.998         |-0.113         |-0.113
srv_serror_rate|-0.031         |-0.001         |-0.001         |+0.005         |-0.007         |-0.001         |-0.012         |-0.002         |-0.189         |-0.001         |-0.003         |-0.001         |-0.001         |-0.004         |-0.004         |-0.013         |NaN            |+0.001         |-0.013         |-0.319         |-0.538         |+0.999         |+1.000         |-0.112         |-0.115         |-0.857         |+0.252         |-0.092         |+0.156         |-0.781         |-0.802         |+0.161         |-0.584         |-0.072         |+0.998         |+0.999         |-0.113         |-0.116
rerror_rate    |+0.017         |+0.003         |+0.002         |-0.000         |-0.004         |-0.000         |-0.006         |+0.006         |-0.099         |-0.000         |-0.001         |-0.000         |-0.001         |-0.002         |-0.002         |-0.007         |NaN            |-0.000         |-0.007         |-0.211         |-0.290         |-0.112         |-0.112         |+1.000         |+0.995         |-0.330         |+0.234         |+0.023         |-0.091         |-0.333         |-0.322         |+0.214         |-0.267         |+0.143         |-0.112         |-0.112         |+0.990         |+0.986
srv_rerror_rate|+0.017         |+0.003         |+0.002         |-0.001         |-0.004         |-0.000         |-0.005         |+0.006         |-0.097         |-0.000         |-0.001         |-0.001         |-0.001         |-0.002         |-0.002         |-0.005         |NaN            |-0.000         |-0.007         |-0.211         |-0.290         |-0.112         |-0.115         |+0.995         |+1.000         |-0.329         |+0.233         |+0.025         |-0.089         |-0.332         |-0.321         |+0.213         |-0.267         |+0.140         |-0.112         |-0.115         |+0.986         |+0.988
same_srv_rate  |+0.022         |+0.001         |+0.001         |+0.001         |+0.006         |+0.001         |+0.013         |+0.002         |+0.217         |+0.001         |+0.004         |+0.002         |+0.002         |+0.005         |+0.004         |+0.015         |NaN            |-0.000         |+0.014         |+0.364         |+0.625         |-0.858         |-0.857         |-0.330         |-0.329         |+1.000         |-0.425         |+0.105         |-0.180         |+0.908         |+0.932         |-0.269         |+0.668         |+0.083         |-0.858         |-0.858         |-0.331         |-0.332
diff_srv_rate  |+0.050         |+0.000         |-0.000         |+0.001         |-0.002         |-0.000         |+0.004         |+0.001         |-0.071         |-0.000         |-0.001         |-0.001         |-0.000         |-0.000         |+0.000         |-0.003         |NaN            |+0.004         |+0.007         |-0.180         |-0.300         |+0.251         |+0.252         |+0.234         |+0.233         |-0.425         |+1.000         |-0.022         |+0.054         |-0.418         |-0.426         |+0.533         |-0.268         |-0.027         |+0.251         |+0.252         |+0.228         |+0.234
srv_diff_host_r|-0.013         |-0.000         |+0.000         |+0.013         |+0.000         |-0.000         |-0.000         |-0.001         |+0.338         |-0.000         |+0.001         |-0.000         |+0.000         |+0.006         |-0.000         |+0.029         |NaN            |-0.000         |-0.003         |-0.314         |-0.235         |-0.092         |-0.092         |+0.023         |+0.025         |+0.105         |-0.022         |+1.000         |-0.384         |+0.007         |+0.047         |-0.007         |-0.189         |+0.256         |-0.091         |-0.093         |+0.024         |+0.020
dst_host_count |+0.011         |-0.002         |-0.002         |-0.009         |-0.002         |-0.003         |-0.036         |-0.010         |-0.628         |-0.005         |-0.011         |-0.008         |-0.007         |-0.017         |-0.014         |-0.019         |NaN            |+0.000         |-0.044         |+0.534         |+0.401         |+0.156         |+0.156         |-0.091         |-0.089         |-0.180         |+0.054         |-0.384         |+1.000         |-0.041         |-0.124         |+0.025         |+0.298         |-0.487         |+0.156         |+0.157         |-0.095         |-0.090
dst_host_srv_co|-0.117         |-0.002         |-0.001         |-0.004         |-0.019         |-0.002         |-0.033         |-0.007         |+0.126         |-0.003         |-0.003         |-0.007         |-0.005         |-0.012         |-0.010         |-0.000         |NaN            |-0.001         |-0.043         |+0.515         |+0.721         |-0.782         |-0.781         |-0.333         |-0.332         |+0.908         |-0.418         |+0.007         |-0.041         |+1.000         |+0.979         |-0.467         |+0.684         |-0.008         |-0.783         |-0.782         |-0.334         |-0.334
dst_host_same_s|-0.119         |-0.002         |-0.001         |+0.001         |-0.017         |-0.002         |-0.025         |-0.003         |+0.157         |-0.002         |-0.000         |-0.005         |-0.004         |-0.009         |-0.008         |+0.005         |NaN            |-0.001         |-0.034         |+0.476         |+0.695         |-0.803         |-0.802         |-0.322         |-0.321         |+0.932         |-0.426         |+0.047         |-0.124         |+0.979         |+1.000         |-0.472         |+0.676         |+0.054         |-0.804         |-0.803         |-0.323         |-0.324
dst_host_diff_s|+0.409         |+0.001         |+0.003         |-0.000         |+0.023         |+0.002         |+0.009         |+0.002         |-0.059         |+0.002         |+0.001         |+0.003         |+0.003         |+0.006         |+0.002         |+0.002         |NaN            |+0.002         |+0.013         |-0.266         |-0.332         |+0.161         |+0.161         |+0.214         |+0.213         |-0.269         |+0.533         |-0.007         |+0.025         |-0.467         |-0.472         |+1.000         |-0.159         |+0.005         |+0.161         |+0.161         |+0.218         |+0.215
dst_host_same_s|+0.043         |-0.001         |-0.001         |+0.001         |-0.010         |-0.001         |-0.032         |-0.005         |-0.461         |-0.002         |-0.009         |-0.005         |-0.003         |-0.010         |-0.004         |-0.034         |NaN            |-0.001         |-0.035         |+0.863         |+0.947         |-0.585         |-0.584         |-0.267         |-0.267         |+0.668         |-0.268         |-0.189         |+0.298         |+0.684         |+0.676         |-0.159         |+1.000         |-0.074         |-0.585         |-0.584         |-0.269         |-0.270
dst_host_srv_di|-0.009         |+0.000         |+0.000         |+0.033         |+0.004         |+0.003         |-0.003         |+0.004         |+0.140         |+0.003         |+0.009         |+0.006         |+0.004         |+0.006         |+0.003         |+0.002         |NaN            |-0.000         |-0.004         |-0.245         |-0.183         |-0.072         |-0.072         |+0.143         |+0.140         |+0.083         |-0.027         |+0.256         |-0.487         |-0.008         |+0.054         |+0.005         |-0.074         |+1.000         |-0.072         |-0.072         |+0.145         |+0.146
dst_host_serror|-0.031         |-0.001         |-0.001         |+0.005         |-0.006         |-0.000         |-0.012         |-0.001         |-0.188         |-0.000         |-0.003         |-0.000         |-0.001         |-0.002         |-0.004         |-0.013         |NaN            |+0.000         |-0.013         |-0.320         |-0.539         |+0.999         |+0.998         |-0.112         |-0.112         |-0.858         |+0.251         |-0.091         |+0.156         |-0.783         |-0.804         |+0.161         |-0.585         |-0.072         |+1.000         |+0.998         |-0.114         |-0.113
dst_host_srv_se|-0.031         |-0.001         |-0.001         |+0.003         |-0.007         |-0.001         |-0.012         |-0.001         |-0.189         |-0.001         |-0.003         |-0.000         |-0.001         |-0.002         |-0.004         |-0.013         |NaN            |+0.001         |-0.013         |-0.319         |-0.538         |+0.998         |+0.999         |-0.112         |-0.115         |-0.858         |+0.252         |-0.093         |+0.157         |-0.782         |-0.803         |+0.161         |-0.584         |-0.072         |+0.998         |+1.000         |-0.113         |-0.116
dst_host_rerror|+0.011         |-0.000         |+0.003         |-0.001         |+0.009         |-0.000         |-0.005         |+0.005         |-0.091         |-0.000         |-0.001         |-0.000         |-0.001         |-0.001         |-0.002         |-0.007         |NaN            |-0.000         |-0.006         |-0.214         |-0.293         |-0.113         |-0.113         |+0.990         |+0.986         |-0.331         |+0.228         |+0.024         |-0.095         |-0.334         |-0.323         |+0.218         |-0.269         |+0.145         |-0.114         |-0.113         |+1.000         |+0.987
dst_host_srv_re|+0.016         |+0.003         |+0.003         |-0.001         |-0.004         |-0.000         |-0.005         |+0.005         |-0.088         |-0.000         |-0.001         |-0.000         |-0.001         |-0.001         |-0.002         |-0.007         |NaN            |-0.000         |-0.006         |-0.212         |-0.291         |-0.113         |-0.116         |+0.986         |+0.988         |-0.332         |+0.234         |+0.020         |-0.090         |-0.334         |-0.324         |+0.215         |-0.270         |+0.146         |-0.113         |-0.116         |+0.987         |+1.000

[srv_serror_rate-dst_host_srv_serror_rate][+0.999]
[serror_rate-dst_host_serror_rate][+0.999]
[serror_rate-srv_serror_rate][+0.999]
[dst_host_serror_rate-dst_host_srv_serror_rate][+0.998]
[serror_rate-dst_host_srv_serror_rate][+0.998]
[srv_serror_rate-dst_host_serror_rate][+0.998]
[num_compromised-num_root][+0.998]
[rerror_rate-srv_rerror_rate][+0.995]
[rerror_rate-dst_host_rerror_rate][+0.990]
[srv_rerror_rate-dst_host_srv_rerror_rate][+0.988]
[dst_host_rerror_rate-dst_host_srv_rerror_rate][+0.987]
[rerror_rate-dst_host_srv_rerror_rate][+0.986]
[srv_rerror_rate-dst_host_rerror_rate][+0.986]
[dst_host_srv_count-dst_host_same_srv_rate][+0.979]
[srv_count-dst_host_same_src_port_rate][+0.947]
[count-srv_count][+0.943]
[same_srv_rate-dst_host_same_srv_rate][+0.932]
[same_srv_rate-dst_host_srv_count][+0.908]
[count-dst_host_same_src_port_rate][+0.863]
[serror_rate-same_srv_rate][-0.858]
[same_srv_rate-dst_host_serror_rate][-0.858]
[same_srv_rate-dst_host_srv_serror_rate][-0.858]
[srv_serror_rate-same_srv_rate][-0.857]
[dst_host_same_srv_rate-dst_host_serror_rate][-0.804]
[hot-is_guest_login][+0.804]
[serror_rate-dst_host_same_srv_rate][-0.803]
[dst_host_same_srv_rate-dst_host_srv_serror_rate][-0.803]
[srv_serror_rate-dst_host_same_srv_rate][-0.802]
[dst_host_srv_count-dst_host_serror_rate][-0.783]
[serror_rate-dst_host_srv_count][-0.782]
[dst_host_srv_count-dst_host_srv_serror_rate][-0.782]
[srv_serror_rate-dst_host_srv_count][-0.781]
[srv_count-dst_host_srv_count][+0.721]
[srv_count-dst_host_same_srv_rate][+0.695]
[dst_host_srv_count-dst_host_same_src_port_rate][+0.684]
[dst_host_same_srv_rate-dst_host_same_src_port_rate][+0.676]
[same_srv_rate-dst_host_same_src_port_rate][+0.668]
[logged_in-count][-0.631]
[logged_in-dst_host_count][-0.628]
[srv_count-same_srv_rate][+0.625]
[dst_host_same_src_port_rate-dst_host_serror_rate][-0.585]
[serror_rate-dst_host_same_src_port_rate][-0.585]
[srv_serror_rate-dst_host_same_src_port_rate][-0.584]
[dst_host_same_src_port_rate-dst_host_srv_serror_rate][-0.584]
[srv_count-dst_host_serror_rate][-0.539]
[srv_count-serror_rate][-0.538]
[srv_count-srv_serror_rate][-0.538]
[srv_count-dst_host_srv_serror_rate][-0.538]
[count-dst_host_count][+0.534]
[diff_srv_rate-dst_host_diff_srv_rate][+0.533]
[count-dst_host_srv_count][+0.515]
[dst_host_count-dst_host_srv_diff_host_rate][-0.487]
[count-dst_host_same_srv_rate][+0.476]
[logged_in-srv_count][-0.473]
[dst_host_same_srv_rate-dst_host_diff_srv_rate][-0.472]
[dst_host_srv_count-dst_host_diff_srv_rate][-0.467]
[logged_in-dst_host_same_src_port_rate][-0.461]
[root_shell-su_attempted][+0.456]
[diff_srv_rate-dst_host_same_srv_rate][-0.426]
[same_srv_rate-diff_srv_rate][-0.425]
[diff_srv_rate-dst_host_srv_count][-0.418]
[duration-dst_host_diff_srv_rate][+0.409]
[srv_count-dst_host_count][+0.401]
[srv_diff_host_rate-dst_host_count][-0.384]
[su_attempted-num_root][+0.378]
[count-same_srv_rate][+0.364]
[num_compromised-su_attempted][+0.350]
[logged_in-srv_diff_host_rate][+0.338]
[dst_host_srv_count-dst_host_srv_rerror_rate][-0.334]
[dst_host_srv_count-dst_host_rerror_rate][-0.334]
[rerror_rate-dst_host_srv_count][-0.333]
[srv_count-dst_host_diff_srv_rate][-0.332]
[same_srv_rate-dst_host_srv_rerror_rate][-0.332]
[srv_rerror_rate-dst_host_srv_count][-0.332]
[same_srv_rate-dst_host_rerror_rate][-0.331]
[rerror_rate-same_srv_rate][-0.330]
[srv_rerror_rate-same_srv_rate][-0.329]
[dst_host_same_srv_rate-dst_host_srv_rerror_rate][-0.324]
[dst_host_same_srv_rate-dst_host_rerror_rate][-0.323]
[rerror_rate-dst_host_same_srv_rate][-0.322]
[srv_rerror_rate-dst_host_same_srv_rate][-0.321]
[count-dst_host_serror_rate][-0.320]
[count-serror_rate][-0.319]
[count-srv_serror_rate][-0.319]
[count-dst_host_srv_serror_rate][-0.319]
[count-srv_diff_host_rate][-0.314]
[srv_count-diff_srv_rate][-0.300]
[dst_host_count-dst_host_same_src_port_rate][+0.298]
[srv_count-dst_host_rerror_rate][-0.293]
[srv_count-dst_host_srv_rerror_rate][-0.291]
[srv_count-rerror_rate][-0.290]
[srv_count-srv_rerror_rate][-0.290]
[dst_host_same_src_port_rate-dst_host_srv_rerror_rate][-0.270]
[dst_host_same_src_port_rate-dst_host_rerror_rate][-0.269]
[same_srv_rate-dst_host_diff_srv_rate][-0.269]
[diff_srv_rate-dst_host_same_src_port_rate][-0.268]
[srv_rerror_rate-dst_host_same_src_port_rate][-0.267]
[rerror_rate-dst_host_same_src_port_rate][-0.267]
[count-dst_host_diff_srv_rate][-0.266]
[su_attempted-num_access_files][+0.259]
[srv_diff_host_rate-dst_host_srv_diff_host_rate][+0.256]
[diff_srv_rate-dst_host_srv_serror_rate][+0.252]
[srv_serror_rate-diff_srv_rate][+0.252]
[diff_srv_rate-dst_host_serror_rate][+0.251]
[serror_rate-diff_srv_rate][+0.251]
[count-dst_host_srv_diff_host_rate][-0.245]
[srv_count-srv_diff_host_rate][-0.235]
[diff_srv_rate-dst_host_srv_rerror_rate][+0.234]
[rerror_rate-diff_srv_rate][+0.234]
[srv_rerror_rate-diff_srv_rate][+0.233]
[diff_srv_rate-dst_host_rerror_rate][+0.228]
[dst_host_diff_srv_rate-dst_host_rerror_rate][+0.218]
[logged_in-same_srv_rate][+0.217]
[dst_host_diff_srv_rate-dst_host_srv_rerror_rate][+0.215]
[rerror_rate-dst_host_diff_srv_rate][+0.214]
[count-dst_host_rerror_rate][-0.214]
[srv_rerror_rate-dst_host_diff_srv_rate][+0.213]
[count-dst_host_srv_rerror_rate][-0.212]
[count-srv_rerror_rate][-0.211]
[count-rerror_rate][-0.211]
[srv_diff_host_rate-dst_host_same_src_port_rate][-0.189]
[logged_in-dst_host_srv_serror_rate][-0.189]
[logged_in-serror_rate][-0.189]
[logged_in-srv_serror_rate][-0.189]
[logged_in-dst_host_serror_rate][-0.188]
[root_shell-num_root][+0.187]
[srv_count-dst_host_srv_diff_host_rate][-0.183]
[count-diff_srv_rate][-0.180]
[same_srv_rate-dst_host_count][-0.180]
[num_compromised-root_shell][+0.172]
[dst_host_diff_srv_rate-dst_host_serror_rate][+0.161]
[serror_rate-dst_host_diff_srv_rate][+0.161]
[dst_host_diff_srv_rate-dst_host_srv_serror_rate][+0.161]
[srv_serror_rate-dst_host_diff_srv_rate][+0.161]
[dst_host_diff_srv_rate-dst_host_same_src_port_rate][-0.159]
[logged_in-dst_host_same_srv_rate][+0.157]
[num_root-num_access_files][+0.157]
[dst_host_count-dst_host_srv_serror_rate][+0.157]
[dst_host_count-dst_host_serror_rate][+0.156]
[serror_rate-dst_host_count][+0.156]
[srv_serror_rate-dst_host_count][+0.156]
[dst_host_srv_diff_host_rate-dst_host_srv_rerror_rate][+0.146]
[dst_host_srv_diff_host_rate-dst_host_rerror_rate][+0.145]
[num_compromised-num_access_files][+0.144]
[rerror_rate-dst_host_srv_diff_host_rate][+0.143]
[srv_rerror_rate-dst_host_srv_diff_host_rate][+0.140]
[logged_in-dst_host_srv_diff_host_rate][+0.140]
[urgent-su_attempted][+0.133]
[root_shell-num_access_files][+0.132]
[logged_in-dst_host_srv_count][+0.126]
[dst_host_count-dst_host_same_srv_rate][-0.124]
[duration-dst_host_same_srv_rate][-0.119]
[duration-dst_host_srv_count][-0.117]
[dst_host_srv_serror_rate-dst_host_srv_rerror_rate][-0.116]
[srv_serror_rate-dst_host_srv_rerror_rate][-0.116]
[srv_serror_rate-srv_rerror_rate][-0.115]
[srv_rerror_rate-dst_host_srv_serror_rate][-0.115]
[dst_host_serror_rate-dst_host_rerror_rate][-0.114]
[serror_rate-dst_host_rerror_rate][-0.113]
[dst_host_srv_serror_rate-dst_host_rerror_rate][-0.113]
[srv_serror_rate-dst_host_rerror_rate][-0.113]
[dst_host_serror_rate-dst_host_srv_rerror_rate][-0.113]
[serror_rate-dst_host_srv_rerror_rate][-0.113]
[serror_rate-rerror_rate][-0.112]
[serror_rate-srv_rerror_rate][-0.112]
[rerror_rate-dst_host_serror_rate][-0.112]
[srv_rerror_rate-dst_host_serror_rate][-0.112]
[srv_serror_rate-rerror_rate][-0.112]
[rerror_rate-dst_host_srv_serror_rate][-0.112]
[duration-count][-0.105]
[same_srv_rate-srv_diff_host_rate][+0.105]
[logged_in-rerror_rate][-0.099]
[logged_in-srv_rerror_rate][-0.097]
[duration-num_file_creations][+0.095]
[dst_host_count-dst_host_rerror_rate][-0.095]
[srv_diff_host_rate-dst_host_srv_serror_rate][-0.093]
[serror_rate-srv_diff_host_rate][-0.092]
[srv_serror_rate-srv_diff_host_rate][-0.092]
[srv_diff_host_rate-dst_host_serror_rate][-0.091]
[logged_in-dst_host_rerror_rate][-0.091]
[rerror_rate-dst_host_count][-0.091]
[dst_host_count-dst_host_srv_rerror_rate][-0.090]
[urgent-root_shell][+0.089]
[srv_rerror_rate-dst_host_count][-0.089]
[logged_in-dst_host_srv_rerror_rate][-0.088]
[same_srv_rate-dst_host_srv_diff_host_rate][+0.083]
[duration-srv_count][-0.080]
[num_file_creations-num_access_files][+0.078]
[dst_host_same_src_port_rate-dst_host_srv_diff_host_rate][-0.074]
[dst_host_srv_diff_host_rate-dst_host_srv_serror_rate][-0.072]
[dst_host_srv_diff_host_rate-dst_host_serror_rate][-0.072]
[serror_rate-dst_host_srv_diff_host_rate][-0.072]
[srv_serror_rate-dst_host_srv_diff_host_rate][-0.072]
[logged_in-diff_srv_rate][-0.071]
[logged_in-is_guest_login][+0.071]
[logged_in-num_access_files][+0.070]
[num_failed_logins-su_attempted][+0.069]
[hot-logged_in][+0.065]
[logged_in-dst_host_diff_srv_rate][-0.059]
[diff_srv_rate-dst_host_count][+0.054]
[dst_host_same_srv_rate-dst_host_srv_diff_host_rate][+0.054]
[duration-su_attempted][+0.052]
[duration-diff_srv_rate][+0.050]
[srv_diff_host_rate-dst_host_same_srv_rate][+0.047]
[is_guest_login-count][-0.046]
[num_access_files-count][-0.045]
[is_guest_login-dst_host_count][-0.044]
[duration-dst_host_same_src_port_rate][+0.043]
[is_guest_login-dst_host_srv_count][-0.043]
[hot-count][-0.042]
[duration-src_bytes][+0.041]
[dst_host_count-dst_host_srv_count][-0.041]
[root_shell-num_shells][+0.037]
[hot-dst_host_count][-0.036]
[root_shell-num_file_creations][+0.035]
[is_guest_login-dst_host_same_src_port_rate][-0.035]
[is_guest_login-srv_count][-0.035]
[num_access_files-dst_host_same_src_port_rate][-0.034]
[num_access_files-srv_count][-0.034]
[is_guest_login-dst_host_same_srv_rate][-0.034]
[hot-dst_host_srv_count][-0.033]
[land-dst_host_srv_diff_host_rate][+0.033]
[hot-srv_count][-0.032]
[hot-dst_host_same_src_port_rate][-0.032]
[duration-srv_serror_rate][-0.031]
[duration-serror_rate][-0.031]
[urgent-num_failed_logins][+0.031]
[urgent-num_root][+0.031]
[duration-dst_host_srv_serror_rate][-0.031]
[duration-dst_host_serror_rate][-0.031]
[num_access_files-srv_diff_host_rate][+0.029]
[duration-num_root][+0.029]
[su_attempted-num_file_creations][+0.028]
[duration-num_compromised][+0.027]
[diff_srv_rate-dst_host_srv_diff_host_rate][-0.027]
[duration-root_shell][+0.026]
[hot-dst_host_same_srv_rate][-0.025]
[dst_host_count-dst_host_diff_srv_rate][+0.025]
[srv_rerror_rate-srv_diff_host_rate][+0.025]
[urgent-num_access_files][+0.024]
[srv_diff_host_rate-dst_host_rerror_rate][+0.024]
[num_failed_logins-root_shell][+0.024]
[duration-num_access_files][+0.024]
[logged_in-num_file_creations][+0.023]
[rerror_rate-srv_diff_host_rate][+0.023]
[wrong_fragment-dst_host_diff_srv_rate][+0.023]
[duration-same_srv_rate][+0.022]
[diff_srv_rate-srv_diff_host_rate][-0.022]
[logged_in-num_shells][+0.021]
[duration-logged_in][-0.021]
[duration-dst_bytes][+0.020]
[logged_in-root_shell][+0.020]
[wrong_fragment-count][-0.020]
[hot-num_file_creations][+0.020]
[srv_diff_host_rate-dst_host_srv_rerror_rate][+0.020]
[num_failed_logins-num_compromised][+0.020]
[wrong_fragment-dst_host_srv_count][-0.019]
[num_access_files-dst_host_count][-0.019]
[num_failed_logins-num_root][+0.019]
[hot-root_shell][+0.018]
[urgent-num_compromised][+0.018]
[wrong_fragment-dst_host_same_srv_rate][-0.017]
[num_file_creations-dst_host_count][-0.017]
[duration-srv_rerror_rate][+0.017]
[duration-rerror_rate][+0.017]
[duration-dst_host_srv_rerror_rate][+0.016]
[wrong_fragment-srv_count][-0.015]
[num_file_creations-count][-0.015]
[num_access_files-same_srv_rate][+0.015]
[num_failed_logins-num_file_creations][+0.014]
[is_guest_login-same_srv_rate][+0.014]
[num_shells-dst_host_count][-0.014]
[land-srv_diff_host_rate][+0.013]
[is_guest_login-dst_host_srv_serror_rate][-0.013]
[is_guest_login-serror_rate][-0.013]
[is_guest_login-srv_serror_rate][-0.013]
[is_guest_login-dst_host_serror_rate][-0.013]
[num_access_files-serror_rate][-0.013]
[num_access_files-srv_serror_rate][-0.013]
[num_shells-count][-0.013]
[hot-same_srv_rate][+0.013]
[num_access_files-dst_host_srv_serror_rate][-0.013]
[duration-srv_diff_host_rate][-0.013]
[root_shell-count][-0.013]
[is_guest_login-dst_host_diff_srv_rate][+0.013]
[num_access_files-dst_host_serror_rate][-0.013]
[hot-dst_host_srv_serror_rate][-0.012]
[hot-srv_serror_rate][-0.012]
[hot-serror_rate][-0.012]
[urgent-num_file_creations][+0.012]
[num_file_creations-dst_host_srv_count][-0.012]
[hot-dst_host_serror_rate][-0.012]
[num_compromised-num_file_creations][+0.012]
[num_file_creations-srv_count][-0.011]
[num_root-num_file_creations][+0.011]
[logged_in-su_attempted][+0.011]
[root_shell-dst_host_count][-0.011]
[duration-dst_host_count][+0.011]
[duration-dst_host_rerror_rate][+0.011]
[num_file_creations-num_shells][+0.010]
[wrong_fragment-dst_host_same_src_port_rate][-0.010]
[num_file_creations-dst_host_same_src_port_rate][-0.010]
[num_shells-srv_count][-0.010]
[num_failed_logins-dst_host_count][-0.010]
[num_shells-dst_host_srv_count][-0.010]
[root_shell-srv_count][-0.010]
[wrong_fragment-dst_host_rerror_rate][+0.009]
[num_file_creations-dst_host_same_srv_rate][-0.009]
[root_shell-dst_host_srv_diff_host_rate][+0.009]
[root_shell-dst_host_same_src_port_rate][-0.009]
[land-dst_host_count][-0.009]
[hot-dst_host_diff_srv_rate][+0.009]
[duration-dst_host_srv_diff_host_rate][-0.009]
[dst_host_srv_count-dst_host_srv_diff_host_rate][-0.008]
[logged_in-num_root][+0.008]
[su_attempted-dst_host_count][-0.008]
[num_shells-dst_host_same_srv_rate][-0.008]
[duration-num_failed_logins][+0.007]
[srv_diff_host_rate-dst_host_srv_count][+0.007]
[su_attempted-count][-0.007]
[wrong_fragment-srv_serror_rate][-0.007]
[wrong_fragment-dst_host_srv_serror_rate][-0.007]
[is_guest_login-rerror_rate][-0.007]
[is_guest_login-diff_srv_rate][+0.007]
[is_guest_login-srv_rerror_rate][-0.007]
[num_failed_logins-dst_host_srv_count][-0.007]
[num_access_files-rerror_rate][-0.007]
[num_failed_logins-count][-0.007]
[su_attempted-dst_host_srv_count][-0.007]
[num_root-dst_host_count][-0.007]
[num_access_files-dst_host_srv_rerror_rate][-0.007]
[srv_diff_host_rate-dst_host_diff_srv_rate][-0.007]
[num_access_files-dst_host_rerror_rate][-0.007]
[is_guest_login-dst_host_srv_rerror_rate][-0.006]
[wrong_fragment-logged_in][-0.006]
[su_attempted-dst_host_srv_diff_host_rate][+0.006]
[wrong_fragment-dst_host_serror_rate][-0.006]
[num_failed_logins-rerror_rate][+0.006]
[num_failed_logins-srv_rerror_rate][+0.006]
[is_guest_login-dst_host_rerror_rate][-0.006]
[hot-rerror_rate][-0.006]
[num_file_creations-dst_host_srv_diff_host_rate][+0.006]
[num_file_creations-srv_diff_host_rate][+0.006]
[num_file_creations-dst_host_diff_srv_rate][+0.006]
[wrong_fragment-same_srv_rate][+0.006]
[hot-srv_rerror_rate][-0.005]
[su_attempted-srv_count][-0.005]
[hot-dst_host_srv_rerror_rate][-0.005]
[su_attempted-dst_host_same_srv_rate][-0.005]
[num_failed_logins-srv_count][-0.005]
[num_root-count][-0.005]
[dst_host_diff_srv_rate-dst_host_srv_diff_host_rate][+0.005]
[land-srv_serror_rate][+0.005]
[logged_in-num_compromised][+0.005]
[hot-dst_host_rerror_rate][-0.005]
[su_attempted-dst_host_same_src_port_rate][-0.005]
[num_failed_logins-dst_host_rerror_rate][+0.005]
[land-serror_rate][+0.005]
[num_failed_logins-dst_host_srv_rerror_rate][+0.005]
[num_root-dst_host_srv_count][-0.005]
[num_access_files-srv_rerror_rate][-0.005]
[num_file_creations-same_srv_rate][+0.005]
[num_failed_logins-dst_host_same_src_port_rate][-0.005]
[num_access_files-dst_host_same_srv_rate][+0.005]
[num_failed_logins-is_guest_login][+0.005]
[land-dst_host_serror_rate][+0.005]
[num_compromised-dst_host_count][-0.005]
[hot-num_failed_logins][+0.004]
[wrong_fragment-serror_rate][-0.004]
[duration-hot][+0.004]
[hot-diff_srv_rate][+0.004]
[root_shell-same_srv_rate][+0.004]
[land-dst_host_srv_count][-0.004]
[num_failed_logins-dst_host_srv_diff_host_rate][+0.004]
[num_shells-same_srv_rate][+0.004]
[wrong_fragment-dst_host_srv_diff_host_rate][+0.004]
[num_file_creations-srv_serror_rate][-0.004]
[num_shells-srv_serror_rate][-0.004]
[num_root-srv_count][-0.004]
[num_root-dst_host_srv_diff_host_rate][+0.004]
[num_file_creations-serror_rate][-0.004]
[num_shells-serror_rate][-0.004]
[num_shells-dst_host_same_src_port_rate][-0.004]
[num_shells-dst_host_serror_rate][-0.004]
[wrong_fragment-dst_host_srv_rerror_rate][-0.004]
[duration-urgent][+0.004]
[wrong_fragment-srv_rerror_rate][-0.004]
[land-count][-0.004]
[num_shells-dst_host_srv_serror_rate][-0.004]
[num_root-dst_host_same_srv_rate][-0.004]
[is_guest_login-dst_host_srv_diff_host_rate][-0.004]
[is_host_login-diff_srv_rate][+0.004]
[wrong_fragment-rerror_rate][-0.004]
[urgent-hot][+0.004]
[su_attempted-dst_host_diff_srv_rate][+0.003]
[is_guest_login-srv_diff_host_rate][-0.003]
[num_shells-dst_host_srv_diff_host_rate][+0.003]
[dst_bytes-dst_host_diff_srv_rate][+0.003]
[num_compromised-count][-0.003]
[src_bytes-srv_rerror_rate][+0.003]
[num_compromised-dst_host_srv_count][-0.003]
[urgent-num_shells][+0.003]
[src_bytes-rerror_rate][+0.003]
[root_shell-serror_rate][-0.003]
[root_shell-srv_serror_rate][-0.003]
[num_compromised-dst_host_srv_diff_host_rate][+0.003]
[land-dst_host_srv_serror_rate][+0.003]
[num_root-dst_host_same_src_port_rate][-0.003]
[num_shells-num_access_files][+0.003]
[root_shell-dst_host_serror_rate][-0.003]
[src_bytes-dst_host_srv_rerror_rate][+0.003]
[root_shell-dst_host_srv_serror_rate][-0.003]
[su_attempted-num_shells][+0.003]
[land-srv_count][-0.003]
[num_root-dst_host_diff_srv_rate][+0.003]
[num_failed_logins-dst_host_same_srv_rate][-0.003]
[urgent-dst_host_count][-0.003]
[root_shell-dst_host_srv_count][-0.003]
[hot-num_compromised][+0.003]
[dst_bytes-count][-0.003]
[hot-dst_host_srv_diff_host_rate][-0.003]
[urgent-logged_in][+0.003]
[num_access_files-diff_srv_rate][-0.003]
[dst_bytes-dst_host_srv_rerror_rate][+0.003]
[urgent-dst_host_srv_diff_host_rate][+0.003]
[num_compromised-srv_count][-0.003]
[dst_bytes-dst_host_rerror_rate][+0.003]
[dst_bytes-srv_rerror_rate][+0.002]
[dst_bytes-rerror_rate][+0.002]
[src_bytes-dst_host_count][-0.002]
[num_file_creations-dst_host_srv_serror_rate][-0.002]
[su_attempted-same_srv_rate][+0.002]
[duration-is_guest_login][+0.002]
[num_shells-dst_host_diff_srv_rate][+0.002]
[wrong_fragment-diff_srv_rate][-0.002]
[num_compromised-dst_host_same_src_port_rate][-0.002]
[num_failed_logins-same_srv_rate][+0.002]
[dst_bytes-logged_in][+0.002]
[num_shells-rerror_rate][-0.002]
[num_shells-srv_rerror_rate][-0.002]
[num_compromised-dst_host_same_srv_rate][-0.002]
[num_shells-dst_host_srv_rerror_rate][-0.002]
[hot-num_root][+0.002]
[num_file_creations-dst_host_serror_rate][-0.002]
[dst_bytes-srv_count][-0.002]
[num_access_files-dst_host_diff_srv_rate][+0.002]
[hot-su_attempted][+0.002]
[urgent-dst_host_srv_count][-0.002]
[num_compromised-dst_host_diff_srv_rate][+0.002]
[urgent-dst_host_diff_srv_rate][+0.002]
[num_failed_logins-srv_serror_rate][-0.002]
[num_failed_logins-logged_in][+0.002]
[wrong_fragment-dst_host_count][-0.002]
[urgent-count][-0.002]
[hot-num_shells][+0.002]
[src_bytes-dst_host_srv_count][-0.002]
[num_root-same_srv_rate][+0.002]
[num_failed_logins-serror_rate][-0.002]
[src_bytes-count][-0.002]
[num_access_files-dst_host_srv_diff_host_rate][+0.002]
[num_shells-dst_host_rerror_rate][-0.002]
[is_host_login-dst_host_diff_srv_rate][+0.002]
[num_failed_logins-dst_host_diff_srv_rate][+0.002]
[logged_in-is_host_login][+0.002]
[urgent-dst_host_same_srv_rate][-0.002]
[src_bytes-dst_host_same_srv_rate][-0.002]
[num_file_creations-srv_rerror_rate][-0.002]
[num_root-is_host_login][+0.002]
[dst_bytes-dst_host_count][-0.002]
[num_file_creations-rerror_rate][-0.002]
[land-dst_host_same_src_port_rate][+0.001]
[num_failed_logins-num_access_files][+0.001]
[num_file_creations-dst_host_rerror_rate][-0.001]
[root_shell-dst_host_srv_rerror_rate][-0.001]
[num_file_creations-is_guest_login][+0.001]
[root_shell-diff_srv_rate][-0.001]
[root_shell-srv_rerror_rate][-0.001]
[hot-is_host_login][+0.001]
[num_compromised-is_host_login][+0.001]
[urgent-srv_count][-0.001]
[dst_bytes-num_compromised][+0.001]
[dst_bytes-num_root][+0.001]
[root_shell-rerror_rate][-0.001]
[root_shell-dst_host_rerror_rate][-0.001]
[dst_bytes-su_attempted][+0.001]
[num_failed_logins-dst_host_srv_serror_rate][-0.001]
[src_bytes-srv_count][-0.001]
[num_root-srv_serror_rate][-0.001]
[num_root-serror_rate][-0.001]
[num_compromised-same_srv_rate][+0.001]
[is_host_login-dst_host_same_srv_rate][-0.001]
[is_host_login-dst_host_srv_count][-0.001]
[dst_bytes-dst_host_srv_count][-0.001]
[urgent-dst_host_same_src_port_rate][-0.001]
[num_root-dst_host_srv_serror_rate][-0.001]
[duration-wrong_fragment][-0.001]
[is_host_login-count][-0.001]
[num_failed_logins-dst_host_serror_rate][-0.001]
[dst_bytes-root_shell][+0.001]
[land-logged_in][-0.001]
[dst_bytes-dst_host_same_srv_rate][-0.001]
[num_file_creations-dst_host_srv_rerror_rate][-0.001]
[land-same_srv_rate][+0.001]
[num_root-dst_host_serror_rate][-0.001]
[dst_bytes-same_srv_rate][+0.001]
[num_root-num_shells][+0.001]
[num_failed_logins-srv_diff_host_rate][-0.001]
[land-dst_host_same_srv_rate][+0.001]
[src_bytes-dst_host_serror_rate][-0.001]
[is_host_login-dst_host_same_src_port_rate][-0.001]
[src_bytes-dst_host_same_src_port_rate][-0.001]
[src_bytes-hot][+0.001]
[root_shell-dst_host_diff_srv_rate][+0.001]
[dst_bytes-serror_rate][-0.001]
[dst_bytes-srv_serror_rate][-0.001]
[dst_bytes-dst_host_serror_rate][-0.001]
[is_host_login-srv_count][-0.001]
[dst_bytes-dst_host_srv_serror_rate][-0.001]
[su_attempted-serror_rate][-0.001]
[src_bytes-dst_host_diff_srv_rate][+0.001]
[is_host_login-dst_host_srv_serror_rate][+0.001]
[num_root-srv_rerror_rate][-0.001]
[num_root-rerror_rate][-0.001]
[src_bytes-same_srv_rate][+0.001]
[su_attempted-srv_serror_rate][-0.001]
[src_bytes-dst_host_srv_serror_rate][-0.001]
[dst_bytes-num_failed_logins][+0.001]
[src_bytes-srv_serror_rate][-0.001]
[num_failed_logins-diff_srv_rate][+0.001]
[num_compromised-num_shells][+0.001]
[land-dst_host_srv_rerror_rate][-0.001]
[urgent-same_srv_rate][+0.001]
[land-srv_rerror_rate][-0.001]
[src_bytes-serror_rate][-0.001]
[hot-num_access_files][+0.001]
[su_attempted-srv_rerror_rate][-0.001]
[num_compromised-srv_serror_rate][-0.001]
[num_compromised-serror_rate][-0.001]
[root_shell-srv_diff_host_rate][+0.001]
[dst_bytes-dst_host_same_src_port_rate][-0.001]
[su_attempted-diff_srv_rate][-0.001]
[land-dst_host_rerror_rate][-0.001]
[num_root-dst_host_srv_rerror_rate][-0.001]
[is_host_login-srv_serror_rate][+0.001]
[num_root-dst_host_rerror_rate][-0.001]
[urgent-serror_rate][-0.001]
[urgent-srv_serror_rate][-0.001]
[urgent-dst_host_srv_serror_rate][-0.001]
[num_compromised-dst_host_srv_serror_rate][-0.001]
[land-diff_srv_rate][+0.001]
[num_compromised-diff_srv_rate][-0.000]
[su_attempted-rerror_rate][-0.000]
[duration-num_shells][-0.000]
[hot-srv_diff_host_rate][-0.000]
[wrong_fragment-is_guest_login][-0.000]
[wrong_fragment-num_access_files][-0.000]
[urgent-dst_host_serror_rate][-0.000]
[num_file_creations-diff_srv_rate][-0.000]
[wrong_fragment-hot][-0.000]
[dst_bytes-diff_srv_rate][-0.000]
[su_attempted-dst_host_serror_rate][-0.000]
[num_compromised-dst_host_serror_rate][-0.000]
[root_shell-dst_host_same_srv_rate][-0.000]
[dst_bytes-num_access_files][+0.000]
[num_compromised-rerror_rate][-0.000]
[land-rerror_rate][-0.000]
[dst_bytes-dst_host_srv_diff_host_rate][+0.000]
[su_attempted-dst_host_srv_rerror_rate][-0.000]
[src_bytes-diff_srv_rate][+0.000]
[num_compromised-dst_host_rerror_rate][-0.000]
[num_compromised-srv_rerror_rate][-0.000]
[num_compromised-srv_diff_host_rate][-0.000]
[dst_bytes-srv_diff_host_rate][+0.000]
[urgent-diff_srv_rate][-0.000]
[urgent-dst_host_rerror_rate][-0.000]
[urgent-dst_host_srv_rerror_rate][-0.000]
[urgent-rerror_rate][-0.000]
[urgent-srv_rerror_rate][-0.000]
[su_attempted-dst_host_rerror_rate][-0.000]
[dst_bytes-num_file_creations][+0.000]
[num_shells-is_guest_login][-0.000]
[num_compromised-dst_host_srv_rerror_rate][-0.000]
[src_bytes-dst_bytes][+0.000]
[root_shell-is_guest_login][-0.000]
[land-dst_host_diff_srv_rate][-0.000]
[urgent-srv_diff_host_rate][-0.000]
[is_host_login-dst_host_count][+0.000]
[su_attempted-dst_host_srv_serror_rate][-0.000]
[src_bytes-logged_in][+0.000]
[su_attempted-srv_diff_host_rate][-0.000]
[is_host_login-dst_host_serror_rate][+0.000]
[dst_bytes-urgent][+0.000]
[is_host_login-dst_host_rerror_rate][-0.000]
[duration-land][-0.000]
[is_host_login-rerror_rate][-0.000]
[is_host_login-srv_rerror_rate][-0.000]
[num_access_files-dst_host_srv_count][-0.000]
[src_bytes-dst_host_rerror_rate][-0.000]
[wrong_fragment-num_file_creations][-0.000]
[src_bytes-srv_diff_host_rate][-0.000]
[su_attempted-is_guest_login][-0.000]
[wrong_fragment-num_shells][-0.000]
[is_host_login-srv_diff_host_rate][-0.000]
[wrong_fragment-srv_diff_host_rate][+0.000]
[dst_bytes-hot][+0.000]
[wrong_fragment-root_shell][-0.000]
[is_host_login-serror_rate][+0.000]
[num_shells-srv_diff_host_rate][-0.000]
[num_shells-diff_srv_rate][+0.000]
[is_host_login-dst_host_srv_diff_host_rate][-0.000]
[num_root-diff_srv_rate][-0.000]
[num_root-is_guest_login][-0.000]
[is_host_login-dst_host_srv_rerror_rate][-0.000]
[land-is_guest_login][-0.000]
[wrong_fragment-su_attempted][-0.000]
[land-num_access_files][-0.000]
[wrong_fragment-num_failed_logins][-0.000]
[is_host_login-same_srv_rate][-0.000]
[land-hot][-0.000]
[num_compromised-is_guest_login][-0.000]
[wrong_fragment-num_root][-0.000]
[num_failed_logins-num_shells][-0.000]
[src_bytes-num_file_creations][+0.000]
[land-wrong_fragment][-0.000]
[src_bytes-is_guest_login][-0.000]
[num_access_files-is_guest_login][-0.000]
[dst_bytes-is_guest_login][+0.000]
[urgent-is_guest_login][-0.000]
[wrong_fragment-num_compromised][-0.000]
[src_bytes-wrong_fragment][-0.000]
[dst_bytes-wrong_fragment][-0.000]
[land-num_file_creations][-0.000]
[src_bytes-num_access_files][-0.000]
[src_bytes-num_shells][+0.000]
[duration-is_host_login][-0.000]
[land-num_shells][-0.000]
[land-root_shell][-0.000]
[is_host_login-is_guest_login][-0.000]
[num_access_files-is_host_login][-0.000]
[wrong_fragment-urgent][-0.000]
[dst_bytes-num_shells][-0.000]
[land-su_attempted][-0.000]
[land-num_failed_logins][-0.000]
[wrong_fragment-is_host_login][-0.000]
[land-num_root][-0.000]
[src_bytes-num_failed_logins][-0.000]
[num_root-srv_diff_host_rate][+0.000]
[num_file_creations-is_host_login][-0.000]
[src_bytes-root_shell][-0.000]
[num_shells-is_host_login][-0.000]
[root_shell-is_host_login][-0.000]
[land-num_compromised][-0.000]
[src_bytes-num_compromised][+0.000]
[src_bytes-dst_host_srv_diff_host_rate][+0.000]
[src_bytes-land][-0.000]
[dst_bytes-is_host_login][+0.000]
[src_bytes-su_attempted][-0.000]
[dst_bytes-land][-0.000]
[su_attempted-is_host_login][-0.000]
[num_failed_logins-is_host_login][-0.000]
[land-urgent][-0.000]
[src_bytes-num_root][-0.000]
[land-is_host_login][-0.000]
[src_bytes-is_host_login][-0.000]
[urgent-is_host_login][-0.000]
[src_bytes-urgent][-0.000]
[num_access_files-num_outbound_cmds][NaN]
[num_outbound_cmds-rerror_rate][NaN]
[num_outbound_cmds-same_srv_rate][NaN]
[dst_bytes-num_outbound_cmds][NaN]
[num_outbound_cmds-dst_host_srv_rerror_rate][NaN]
[num_outbound_cmds-is_guest_login][NaN]
[num_file_creations-num_outbound_cmds][NaN]
[wrong_fragment-num_outbound_cmds][NaN]
[num_outbound_cmds-srv_rerror_rate][NaN]
[urgent-num_outbound_cmds][NaN]
[num_outbound_cmds-serror_rate][NaN]
[num_outbound_cmds-dst_host_srv_serror_rate][NaN]
[num_outbound_cmds-count][NaN]
[num_outbound_cmds-srv_serror_rate][NaN]
[num_outbound_cmds-dst_host_same_srv_rate][NaN]
[num_compromised-num_outbound_cmds][NaN]
[num_outbound_cmds-dst_host_srv_diff_host_rate][NaN]
[num_outbound_cmds-dst_host_srv_count][NaN]
[num_outbound_cmds-diff_srv_rate][NaN]
[root_shell-num_outbound_cmds][NaN]
[num_outbound_cmds-dst_host_rerror_rate][NaN]
[num_root-num_outbound_cmds][NaN]
[logged_in-num_outbound_cmds][NaN]
[src_bytes-num_outbound_cmds][NaN]
[num_outbound_cmds-srv_count][NaN]
[land-num_outbound_cmds][NaN]
[num_outbound_cmds-dst_host_same_src_port_rate][NaN]
[num_outbound_cmds-is_host_login][NaN]
[num_shells-num_outbound_cmds][NaN]
[num_outbound_cmds-dst_host_serror_rate][NaN]
[num_failed_logins-num_outbound_cmds][NaN]
[duration-num_outbound_cmds][NaN]
[su_attempted-num_outbound_cmds][NaN]
[num_outbound_cmds-dst_host_diff_srv_rate][NaN]
[num_outbound_cmds-srv_diff_host_rate][NaN]
[num_outbound_cmds-dst_host_count][NaN]
[hot-num_outbound_cmds][NaN]
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
    val multiSelect: Seq[(Int, (Int, Double))] = DTUtil.selFeaturesbyRfClassifier(lps, numClasses)
    multiSelect.
      map{ case (id, (cnt, avg)) => (id.toString.toInt, (cnt.toString.toInt, avg))}.
      foreach{ case (k, (cnt, avg)) => println(f"feature=${k}%-2s(${names.getOrElse(k, "NA")}%-20.20s), cnt=${cnt}%6d, avg.gain=${avg}%8.5f") }
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
        map{ case (id, (cnt, gain)) => id.toString.toInt } // selectFeatureIds.size = 10
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
