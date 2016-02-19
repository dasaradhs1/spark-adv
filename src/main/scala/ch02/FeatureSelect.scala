package ch02

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vector, Matrices, Matrix, Vectors}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.mllib.regression.{RidgeRegressionWithSGD, LassoWithSGD, LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.{Node, DecisionTreeModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import tw.com.chttl.spark.mllib.util.{DTUtil,LRUtil}

import scala.collection.mutable


/**
 * Created by leorick on 2016/2/17.
 */
object FeatureSelect {

  /**
   * ref 20141102 Selecting good features – Part I: univariate selection
   * @param sc
   */
  def correlation(sc:SparkContext) = {
    val rows = 30000000L
    val partions = 2
    val x1 = RandomRDDs.normalRDD(sc, rows, partions, 100L).map{v => 0 + 1 * v}
    /*
    val stats = Statistics.colStats(x1.map{Vectors.dense(_)})
    println(f"${stats.mean}%s, ${stats.min}%s, ${stats.max}%s, ${stats.variance}%s")
[1.1135286200832187E-4], [-5.5773546764577295], [5.752555687223046], [0.9999000344860648]
     */
    val y1 = x1.zip( RandomRDDs.normalRDD(sc, rows, partions, 200L).map{v => 0 + 1 * v} ).
      map{ case (x,y) => x + y}
    /*
    val stats = Statistics.colStats(y1.map{Vectors.dense(_)})
    println(f"${stats.mean}%s, ${stats.min}%s, ${stats.max}%s, ${stats.variance}%s")
[-4.83395309329399E-5], [-8.023287269790373], [8.045078969043846], [2.0000831184953913]
     */
    val y2 = x1.zip( RandomRDDs.normalRDD(sc, rows, partions, 300L).map{v => 0 + 10 * v} ).
      map{ case (x,y) => x + y}
    /*
    val stats = Statistics.colStats(y2.map{Vectors.dense(_)})
    println(f"${stats.mean}%s, ${stats.min}%s, ${stats.max}%s, ${stats.variance}%s")
[-0.003396933111389483], [-54.66028883299299], [54.422030793670906], [100.98488791606434]
     */
    Statistics.corr(x1, y1, "pearson") // = 0.7071464386034881
    Statistics.corr(x1, y2, "pearson") // = 0.0992873024390485

    val x2 = RandomRDDs.uniformRDD(sc, rows, partions, 100L).map{v => -1 + (1 - (-1)) * v}
    /*
    val stats = Statistics.colStats(x2.map{Vectors.dense(_)})
    println(f"${stats.mean}%s, ${stats.min}%s, ${stats.max}%s, ${stats.variance}%s")
[1.065852410082148E-4], [-0.999999973548773], [0.9999999597681517], [0.3334120470732483]

    Statistics.corr(x2, x2.map{x => x*x}, "pearson") = 3.205046433331225E-4
     */
  }

  /**
   * Independent variables
   * ref: 20141112 Selecting good features – Part II: linear models and regularization
   * @param sc
   */
  def LrIndependent(sc:SparkContext) = {
    val cols = 3
    val rows = 30000000L
    val partions = 2
    val x1: RDD[Vector] = RandomRDDs.normalVectorRDD(sc, rows, cols, partions, 100L)
    /*
    val stats = Statistics.colStats(x1)
    println(f"${stats.mean}%s, ${stats.min}%s, ${stats.max}%s, ${stats.variance}%s")
[-1.055723744941577E-4,1.4041202918277184E-4,-7.967544618865945E-5],
[-5.877471597099673   ,-5.6488187080982675  ,-5.142338742306299],
[5.579155580862173    ,5.752555687223046    ,5.877737125992856],
[0.9999499876078658   ,0.9996836029535467   ,1.0000667261424816]
     */
    // Y1 = X0 + 2*X1 + noise, X0 and X1 are independent
    val y1: RDD[Double] = x1.zip( RandomRDDs.normalRDD(sc, rows, partions, 200L).map{v => 0 + 1 * v} ).
      map{ case (x, n) => x(0) + 2*x(1) + n }
    /*
    val stats = Statistics.colStats(y1.map{Vectors.dense(_)})
    println(f"${stats.mean}%s, ${stats.min}%s, ${stats.max}%s, ${stats.variance}%s")
[1.555929093010766E-5], [-14.768514111780256], [14.141883223882898], [5.998153998030806]
     */
    val dataTrain1: RDD[LabeledPoint] = y1.zip(x1).map{ case (y, x) => new LabeledPoint(y, x)}
    dataTrain1.cache()
    val modelLr1 = (new LinearRegressionWithSGD()).run(dataTrain1)
    /*
     printLrModel(modelLr1.weights)
(+0.99987) * X0 + (+1.99975) * X1 + (+0.00043) * X2
     */
  }

  /**
   * Dependent variables
   * ref: 20141112 Selecting good features – Part II: linear models and regularization
   * @param sc
   */
  def LrDependent(sc:SparkContext) = {
    val rows = 100L
    val partions = 2
    val x20: RDD[Double] = RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 1 * v}
    // X1 = X0 + noise
    val x21 = x20.zip( RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 0.1 * v} ).
      map{ case (x: Double, n: Double) => x+n }
    val x22 = x20.zip( RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 0.1 * v} ).
      map{ case (x: Double, n: Double) => x+n }
    val x23 = x20.zip( RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 0.1 * v} ).
      map{ case (x: Double, n: Double) => x+n }
    // Y2 = X1 + X2 + X3 + noise, X1,X2,X3 are dependent
    val y2: RDD[Double] = x21.
      zip{x22}.map{ case (x: Double,y: Double) => x+y }.
      zip{x23}.map{ case (x: Double,y: Double) => x+y }.
      zip{ RandomRDDs.normalRDD(sc, rows, partions, 100L).map{v => 0 + 1 * v} }.
      map{ case (x: Double,y: Double) => x+y }
    /*
    printStats(y2.map{Vectors.dense(_)})
     */
    val dataTrain2 = x21.
      zip{x22}.map{ case (x: Double,y: Double) => Array(x,y) }.
      zip{x23}.map{ case (ary: Array[Double], x1: Double) => ary :+ x1 }.
      zip{y2}.map{ case (ary: Array[Double], y: Double) => new LabeledPoint(y, Vectors.dense(ary)) }
    dataTrain2.cache()
    val modelLr2 = (new LinearRegressionWithSGD()).run(dataTrain2)
    /*
    printLrModel(modelLr2.weights)
(+1.01336) * X0 + (+0.79253) * X1 + (+1.24435) * X2 , rows = 100L
(+1.01530) * X0 + (+1.01915) * X1 + (+0.99329) * X2 , rows = 1000L
(+0.99997) * X0 + (+0.99991) * X1 + (+1.00005) * X2 , rows = 30000000L
隨著資料量增加, 模型參數較不會受到變數相依性影響
     */
    val scaler = new StandardScaler(true, true)
    val modelScale = scaler.fit(dataTrain2.map{ lp => lp.features })
    val dataScale = modelScale.transform(dataTrain2.map{ lp => lp.features }).
      zip(dataTrain2).
      map{ case (v: Vector, lp: LabeledPoint) => new LabeledPoint(lp.label, v) }
    dataScale.cache()
    val iter = 100
    val step = 1.0
    val miniBatch = 1.0
    val modelLasso1 = LassoWithSGD.train(dataScale, iter, step, 0.3, miniBatch)
    /*
    printLrModel(modelLasso1.weights)
(+0.75163) * X0 + (+0.87870) * X1 + (+1.25860) * X2
Lasso模型仍會受到變數相依性影響
     */
  }

  /**
   * L1 regularization model
   * ref: 20141112 Selecting good features – Part II: linear models and regularization
   * @param sc
   */
  def L1(sc:SparkContext) = {
    val src = sc.textFile("file:///home/leoricklin/dataset/boston/housing.data")
    val toks = src.map{ line => line.trim.split("""\s+""").map{_.toDouble} }
    // toks.map{ ary => ary.size }.countByValue() =  Map(14 -> 506)
    val data = toks.
      map{ ary => new LabeledPoint(ary.last, Vectors.dense(ary.slice(0,13))) }
    val scaler = new StandardScaler(true, true)
    val modelScale = scaler.fit(data.map{ lp => lp.features })
    val dataScale = modelScale.transform(data.map{ lp => lp.features }).
      zip(data).
      map{ case (v: Vector, lp: LabeledPoint) => new LabeledPoint(lp.label, v) }
    dataScale.cache()
    val iter = 100
    val step = 1.0
    val miniBatch = 1.0
    val modelLasso1 = LassoWithSGD.train(dataScale, iter, step, 0.3, miniBatch)
    /*
    printLrModel(modelLasso1.weights, Some(Array("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT")), true)
(-3.70770) * LSTAT + (+2.96273) * RM + (-1.74571) * PTRATIO + (-1.27707) * DIS + (-0.83315) * NOX + (+0.61511) * B + (+0.54096) * CHAS + (-0.26205) * CRIM + (+0.17593) * ZN + (-0.00000) * TAX + (+0.00000) * RAD + (-0.00000) * AGE + (-0.00000) * INDUS
-3.707 * LSTAT     + 2.992 * RM      + -1.757 * PTRATIO     + -1.081 * DIS     + -0.7 * NOX       + 0.631 * B      + 0.54 * CHAS       + -0.236 * CRIM     + 0.081 * ZN      + -0.0 * INDUS     + -0.0 * AGE       + 0.0 * RAD        + -0.0 * TAX
     */
    val modelLasso2 = LassoWithSGD.train(dataScale, iter, step, 0.6, miniBatch)
    /*
    printLrModel(modelLasso2.weights, Some(Array("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT")), true)
(-3.61177) * LSTAT + (+2.92386) * RM + (-1.55384) * PTRATIO + (+0.46232) * B + (+0.32846) * CHAS + (-0.05152) * CRIM + (-0.01107) * DIS + (-0.00000) * TAX + (-0.00000) * RAD + (-0.00000) * AGE + (-0.00000) * NOX + (-0.00000) * INDUS + (+0.00000) * ZN
-3.592 * LSTAT     + 2.937 * RM      + -1.544 * PTRATIO     + 0.47 * B       + 0.336 * CHAS      + -0.034 * CRIM     + 0.0 * ZN + -0.0 * INDUS + -0.0 * NOX + -0.0 * AGE + -0.0 * DIS + -0.0 * RAD + -0.0 * TAX
     */
  }

  /**
   * L2 regularization model
   * ref: 20141112 Selecting good features – Part II: linear models and regularization
   * @param sc
   */
  def L2(sc:SparkContext) = {
    val rows = 100L
    val partions = 2
    val out = for (seed <- 1 to 10) yield {
      val x20: RDD[Double] = RandomRDDs.normalRDD(sc, rows, partions, seed).map{v => 0 + 1 * v}
      // X1 = X0 + noise
      val x21 = x20.zip( RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 0.1 * v} ).
        map{ case (x: Double, n: Double) => x+n }
      val x22 = x20.zip( RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 0.1 * v} ).
        map{ case (x: Double, n: Double) => x+n }
      val x23 = x20.zip( RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 0.1 * v} ).
        map{ case (x: Double, n: Double) => x+n }
      // Y2 = X1 + X2 + X3 + noise, X1,X2,X3 are dependent
      val y2: RDD[Double] = x21.
        zip{x22}.map{ case (x: Double,y: Double) => x+y }.
        zip{x23}.map{ case (x: Double,y: Double) => x+y }.
        zip{ RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 1 * v} }.
        map{ case (x: Double,y: Double) => x+y }
      val dataTrain = x21.
        zip{x22}.map{ case (x: Double,y: Double) => Array(x,y) }.
        zip{x23}.map{ case (ary: Array[Double], x1: Double) => ary :+ x1 }.
        zip{y2}.map{ case (ary: Array[Double], y: Double) => new LabeledPoint(y, Vectors.dense(ary)) }
      dataTrain.cache()
      val iter = 100
      val modelLr = LinearRegressionWithSGD.train(dataTrain, iter)
      val modelRidge = RidgeRegressionWithSGD.train(dataTrain, iter)
      f"Random Seed ${seed}" + "\n" +
        "Liner Model: " + LRUtil.printLrModel(modelLr.weights) + "\n" +
        "Ridge Model: " + LRUtil.printLrModel(modelRidge.weights) + "\n"
    }
    println(out.mkString(""))
    /*
Random Seed 1
Liner Model: (+1.09289) * X0 + (+0.95569) * X1 + (+0.77183) * X2
Ridge Model: (+1.06581) * X0 + (+0.95032) * X1 + (+0.79772) * X2
Random Seed 2
Liner Model: (+1.05171) * X0 + (+1.02643) * X1 + (+1.04814) * X2
Ridge Model: (+1.04789) * X0 + (+1.02117) * X1 + (+1.04344) * X2
Random Seed 3
Liner Model: (+0.95749) * X0 + (+0.95393) * X1 + (+0.99433) * X2
Ridge Model: (+0.95626) * X0 + (+0.95201) * X1 + (+0.98757) * X2
Random Seed 4
Liner Model: (+0.91906) * X0 + (+0.95282) * X1 + (+0.93008) * X2
Ridge Model: (+0.91618) * X0 + (+0.94966) * X1 + (+0.92710) * X2
Random Seed 5
Liner Model: (+1.02776) * X0 + (+1.01463) * X1 + (+1.01538) * X2
Ridge Model: (+1.02453) * X0 + (+1.01205) * X1 + (+1.01229) * X2
Random Seed 6
Liner Model: (+0.97821) * X0 + (+1.09051) * X1 + (+0.95023) * X2
Ridge Model: (+0.97874) * X0 + (+1.07875) * X1 + (+0.95283) * X2
Random Seed 7
Liner Model: (+0.93717) * X0 + (+0.93791) * X1 + (+0.94462) * X2
Ridge Model: (+0.93471) * X0 + (+0.93547) * X1 + (+0.94201) * X2
Random Seed 8
Liner Model: (+1.11127) * X0 + (+0.87162) * X1 + (+1.03471) * X2
Ridge Model: (+1.09107) * X0 + (+0.88944) * X1 + (+1.02820) * X2
Random Seed 9
Liner Model: (+0.74883) * X0 + (+1.24046) * X1 + (+1.07787) * X2
Ridge Model: (+0.79335) * X0 + (+1.20094) * X1 + (+1.06465) * X2
Random Seed 10
Liner Model: (+1.06637) * X0 + (+0.98725) * X1 + (+1.07968) * X2
Ridge Model: (+1.06084) * X0 + (+0.98896) * X1 + (+1.07095) * X2
     */
  }

  /**
   * using RF model to select features of  Boston dataset
   * ref: 20141201 Selecting good features – Part III: random forests
   * @param sc
   */
  def Rf4Boston(sc:SparkContext) = {
    val src: RDD[String] = sc.textFile("file:///home/leoricklin/dataset/boston/housing.data", 2)
    val dataTrain: RDD[LabeledPoint] = src.map{ line => line.trim.split("""\s+""").map{_.toDouble} }.
      map{ ary => new LabeledPoint(ary.last, Vectors.dense(ary.slice(0,13))) }
    dataTrain.cache
    val subtree = 10
    val depth = 4
    val bins = 100
    val modelRf = RandomForest.trainRegressor(dataTrain, Map(3 -> 2), subtree, "auto","variance", depth, bins, 100)
    /*
 Tree 0:
    If (feature 4 <= 0.668)
     If (feature 5 <= 6.939)
      If (feature 12 <= 9.62)
       If (feature 7 <= 1.3861)
        Predict: 50.0
       Else (feature 7 > 1.3861)
        Predict: 25.150000000000006
      Else (feature 12 > 9.62)
       If (feature 12 <= 19.78)
        Predict: 19.994535519125684
       Else (feature 12 > 19.78)
        Predict: 15.213043478260854
     Else (feature 5 > 6.939)
      If (feature 10 <= 14.9)
       If (feature 5 <= 7.333)
        Predict: 35.288888888888884
       Else (feature 5 > 7.333)
        Predict: 49.170588235294126
      Else (feature 10 > 14.9)
       If (feature 5 <= 7.42)
        Predict: 33.5030303030303
       Else (feature 5 > 7.42)
        Predict: 43.08999999999996
    Else (feature 4 > 0.668)
     If (feature 0 <= 6.80117)
      If (feature 5 <= 4.368)
       Predict: 27.5
      Else (feature 5 > 4.368)
       If (feature 6 <= 84.4)
        Predict: 21.416666666666668
       Else (feature 6 > 84.4)
        Predict: 16.505714285714284
     Else (feature 0 > 6.80117)
      If (feature 0 <= 15.8603)
       If (feature 12 <= 23.34)
        Predict: 13.563636363636363
       Else (feature 12 > 23.34)
        Predict: 10.64615384615385
      Else (feature 0 > 15.8603)
       If (feature 5 <= 5.701)
        Predict: 7.546153846153846
       Else (feature 5 > 5.701)
        Predict: 11.350000000000001
     */
    val names = Map(0->"CRIM", 1->"ZN", 2->"INDUS", 3->"CHAS", 4->"NOX"
      , 5->"RM", 6->"AGE", 7->"DIS", 8->"RAD", 9->"TAX"
      , 10->"PTRATIO", 11->"B", 12->"LSTAT", 13->"MEDV")
    /*
    // compute 1st subtree
    nodeAryAvgGain(DTUtil.dTree2Array(modelRf.trees(0))).
      toArray.sortBy{ case (k, (cnt, avg)) => avg}.reverse.
      foreach{ case (k, (cnt, avg)) => println(f"feature=${names(k)}%10s, cnt=${cnt}%6d, avg.gain=${avg}%8.5f") }
feature=        RM, cnt=     5, avg.gain=21.68072
feature=       DIS, cnt=     1, avg.gain=20.83674
feature=   PTRATIO, cnt=     1, avg.gain=17.50047
feature=       NOX, cnt=     1, avg.gain=17.28201
feature=      CRIM, cnt=     2, avg.gain= 7.47626
feature=     LSTAT, cnt=     3, avg.gain= 4.88566
feature=       AGE, cnt=     1, avg.gain= 3.01289
    // compute 2nd subtree
    nodeAryAvgGain(DTUtil.dTree2Array(modelRf.trees(1))).
      toArray.sortBy{ case (k, (cnt, avg)) => avg}.reverse.
      foreach{ case (k, (cnt, avg)) => println(f"feature=${names(k)}%10s, cnt=${cnt}%6d, avg.gain=${avg}%8.5f") }
feature=     LSTAT, cnt=     4, avg.gain=51.92247
feature=        RM, cnt=     3, avg.gain=25.65652
feature=     INDUS, cnt=     2, avg.gain=12.78005
feature=      CRIM, cnt=     1, avg.gain= 9.09854
feature=       DIS, cnt=     3, avg.gain= 5.94824
    // compute 3rd subtree
    nodeAryAvgGain(DTUtil.dTree2Array(modelRf.trees(2))).
      toArray.sortBy{ case (k, (cnt, avg)) => avg}.reverse.
      foreach{ case (k, (cnt, avg)) => println(f"feature=${names(k)}%10s, cnt=${cnt}%6d, avg.gain=${avg}%8.5f") }
feature=      CRIM, cnt=     2, avg.gain=34.41788
feature=        RM, cnt=     4, avg.gain=32.49319
feature=   PTRATIO, cnt=     2, avg.gain=13.32143
feature=     LSTAT, cnt=     2, avg.gain= 8.85268
feature=       DIS, cnt=     1, avg.gain= 7.81466
feature=       AGE, cnt=     2, avg.gain= 3.63085
feature=       RAD, cnt=     1, avg.gain= 2.96416
feature=         B, cnt=     1, avg.gain= 1.14800
    // compute avg of subtree(1,2)
    nodeTreeAvgGain(modelRf.trees.take(3)).
      toArray.sortBy{ case (k, (cnt, avg)) => avg}.reverse.
      foreach{ case (k, (cnt, avg)) => println(f"feature=${names(k)}%10s, cnt=${cnt}%6d, avg.gain=${avg}%8.5f") }
feature=     LSTAT, cnt=     9, avg.gain=26.67247
feature=        RM, cnt=    12, avg.gain=26.27883
feature=      CRIM, cnt=     5, avg.gain=18.57737
feature=       NOX, cnt=     1, avg.gain=17.28201
feature=   PTRATIO, cnt=     3, avg.gain=14.71444
feature=     INDUS, cnt=     2, avg.gain=12.78005
feature=       DIS, cnt=     5, avg.gain= 9.29922
feature=       AGE, cnt=     3, avg.gain= 3.42486
feature=       RAD, cnt=     1, avg.gain= 2.96416
feature=         B, cnt=     1, avg.gain= 1.14800
     */
    DTUtil.nodeTreeAvgGain(modelRf.trees).
      toArray.sortBy{ case (k, (cnt, avg)) => avg}.reverse.
      foreach{ case (k, (cnt, avg)) => println(f"feature=${names(k)}%10s, cnt=${cnt}%6d, avg.gain=${avg}%8.5f") }
    /*
feature=        RM, cnt=    39, avg.gain=24.90947
feature=     LSTAT, cnt=    28, avg.gain=23.16337
feature=       DIS, cnt=    13, avg.gain=17.20506
feature=      CRIM, cnt=    13, avg.gain=10.71813
feature=   PTRATIO, cnt=    12, avg.gain=10.56961
feature=       NOX, cnt=     9, avg.gain=10.23514
feature=       AGE, cnt=     8, avg.gain=10.19981
feature=       RAD, cnt=     5, avg.gain= 9.93455
feature=       TAX, cnt=     4, avg.gain= 9.70138
feature=     INDUS, cnt=     3, avg.gain= 8.93830
feature=         B, cnt=     3, avg.gain= 1.70932
feature=        ZN, cnt=     2, avg.gain= 1.31288

    // the result from reference
[(0.4264, 'LSTAT'), (0.3805, 'RM'), (0.068, 'DIS'), (0.0367, 'CRIM'), (0.0248, 'PTRATIO'),
 (0.0213, 'NOX'), (0.0131, 'AGE'), (0.0119, 'TAX'), (0.0066, 'INDUS'), (0.0064, 'B'),
 (0.0031, 'RAD'), (0.0008, 'ZN'), (0.0003, 'CHAS')]
     */
  }

  /**
   * using RF model to select dependent features
   * ref: 20141201 Selecting good features – Part III: random forests
   * @param sc
   */
  def Rf4Dependent(sc:SparkContext) = {
    val rows = 100L
    val partions = 2
    val seed = 100L
    val x0: RDD[Double] = RandomRDDs.normalRDD(sc, rows, partions, seed).map{v => 0 + 1 * v}
    // X1 = X0 + noise
    val x1 = x0.zip( RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 0.1 * v} ).
      map{ case (x: Double, n: Double) => x+n }
    val x2 = x0.zip( RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 0.1 * v} ).
      map{ case (x: Double, n: Double) => x+n }
    val x3 = x0.zip( RandomRDDs.normalRDD(sc, rows, partions).map{v => 0 + 0.1 * v} ).
      map{ case (x: Double, n: Double) => x+n }
    // Y = X1 + X2 + X3, X1,X2,X3 are dependent
    val y: RDD[Double] = x1.
      zip{x2}.map{ case (x: Double,y: Double) => x+y }.
      zip{x3}.map{ case (x: Double,y: Double) => x+y }
    val dataTrain = x1.
      zip{x2}.map{ case (x: Double,y: Double) => Array(x,y) }.
      zip{x3}.map{ case (ary: Array[Double], x1: Double) => ary :+ x1 }.
      zip{y}.map{ case (ary: Array[Double], y: Double) => new LabeledPoint(y, Vectors.dense(ary)) }
    dataTrain.cache()
    val subtree = 10
    val depth = 4
    val bins = 100
    val modelRf = RandomForest.trainRegressor(dataTrain, Map[Int,Int](), subtree, "auto","variance", depth, bins, 100)
    val names = Map(0->"X1", 1->"X2", 2->"X3")
    DTUtil.nodeTreeAvgGain(modelRf.trees).
      toArray.sortBy{ case (k, (cnt, avg)) => avg}.reverse.
      foreach{ case (k, (cnt, avg)) => println(f"feature=${names(k)}%10s, cnt=${cnt}%6d, avg.gain=${avg}%8.5f") }
    /*
feature=        X3, cnt=    71, avg.gain= 0.89168
feature=        X2, cnt=    79, avg.gain= 0.75441
     */

  }

  def featureImpact4Boston(sc:SparkContext) = {
    val src: RDD[String] = sc.textFile("file:///home/leoricklin/dataset/boston/housing.data", 2)
    val dataTrain: RDD[LabeledPoint] = src.map{ line => line.trim.split("""\s+""").map{_.toDouble} }.
      map{ ary => new LabeledPoint(ary.last, Vectors.dense(ary.slice(0,13))) }
    val numFolds = 4
    val seed = 100
    // dataTrain.count = 100
    val folds: Array[(RDD[LabeledPoint], RDD[LabeledPoint])] = MLUtils.kFold(dataTrain, numFolds, seed)
    folds.foreach{ case (train, test) => println(Array(train.count(), test.count()).mkString(",")) }

  }

  def main(args: Array[String]) {
    val appName = "Spam classification"
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)


  }

}
