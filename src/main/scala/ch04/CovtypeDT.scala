package ch04

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.tree.{RandomForest, DecisionTree}
import org.apache.spark.mllib.tree.model.{Node, RandomForestModel, DecisionTreeModel}
import org.apache.spark.rdd._
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import tw.com.chttl.spark.mllib.util.DTUtil
import tw.com.chttl.spark.test.util._
import java.io._

/**
 * Created by leorick on 2015/8/11.
 */
object CovtypeDT {
  val appName = "Basic Decision Tree for Covtype classification"

  def oneHotSimpleDt(sc:SparkContext,  args: Array[String]) = {
    val Array(inpath) = args
    val dataLabeled = DTreeUtil.covtype2OneHotFeature(sc, inpath)
    val Array(dataTrain, dataCV, dataTest) = dataLabeled.randomSplit(Array(0.8, 0.1, 0.1))
    dataTrain.cache();dataCV.cache();dataTest.cache()
    /*
    dataRaw.count   = 581012

    labeledData.map{lp => lp.label}.countByValue() : Map[Double,Long]
     = Map(0.0 -> 211840, 1.0 -> 283301, 2.0 -> 35754, 3.0 -> 2747
     , 4.0 -> 9493, 5.0 -> 17367, 6.0 -> 20510)

    dataTrain.count = 465356
    dataTrain.map{lp => lp.label}.countByValue() : Map[Double,Long]
     = Map(0.0 -> 169767, 1.0 -> 226908, 2.0 -> 28528, 3.0 -> 2212
     , 4.0 -> 7588, 5.0 -> 13885, 6.0 -> 16468)

    dataCV.count    = 57571
    dataCV.map{lp => lp.label}.countByValue(): Map[Double,Long]
     = Map(0.0 -> 20895, 1.0 -> 28168, 2.0 -> 3591, 3.0 -> 268
     , 4.0 -> 918, 5.0 -> 1709, 6.0 -> 2022)

    dataTest.count  = 58085
    dataTest.map{lp => lp.label}.countByValue(): Map[Double,Long]
     = Map(0.0 -> 21178, 1.0 -> 28225, 2.0 -> 3635, 3.0 -> 267
     , 4.0 -> 987, 5.0 -> 1773, 6.0 -> 2020)
     */

    /* (precision, recall) for each category
    DTreeUtil.simpleDecisionTree(dataTrain, dataCV)
    (0.677584541062802   ,0.6712610672409668)
    (0.7240188894316887  ,0.7892289122408407)
    (0.6370386084005091  ,0.8362573099415205)
    (0.46218487394957986 ,0.41044776119402987)
    (0.0,0.0)
    (0.726027397260274   ,0.0310122878876536)
    (0.7274320771253286  ,0.410484668644906)
     */
    val model: DecisionTreeModel = DecisionTree.trainClassifier(dataTrain, 7, Map[Int,Int](), "gini", 4, 100)
    val metrics: MulticlassMetrics = DTreeUtil.getMetrics(model, dataCV)
    /*
    metrics.precision: Double = 0.6996101063190258
    metrics.confusionMatrix: Matrix =
     14248.0  6615.0   5.0     0.0    0.0  1.0   422.0
     5556.0   22440.0  355.0   19.0   0.0  4.0   41.0
     0.0      452.0    3050.0  74.0   0.0  14.0  0.0
     0.0      0.0      163.0   109.0  0.0  0.0   0.0
     0.0      885.0    40.0    1.0    0.0  0.0   0.0
     0.0      564.0    1091.0  37.0   0.0  53.0  0.0
     1078.0   24.0     0.0     0.0    0.0  0.0   883.0

    metricsCV.labels.map{ i => (metricsCV.precision(i), metricsCV.recall(i)) }.foreach(println)
     = Array(
       (0.0,0.36655158791501347),  (1.0,0.48406932206592124), (2.0,0.06260627608594838)
     , (3.0,0.004929492794696072), (4.0,0.01597361776678518), (5.0,0.029954827295992855)
     , (6.0,0.03591487607564281))

    val randomGuess: Array[(Double, Double)] = DTreeUtil.classProbabilities(dataCV)
     */
    Array((model, ("gini", 4, 100), metrics))
    //
  }

  def oneHotMultiParamDt(sc:SparkContext,  args: Array[String]) = {
    val Array(inpath) = args
    val dataLabeled = DTreeUtil.covtype2OneHotFeature(sc, inpath)
    val Array(dataTrain, dataCV, dataTest) = dataLabeled.randomSplit(Array(0.8, 0.1, 0.1))
    dataTrain.cache();dataCV.cache();dataTest.cache()
    val evaluations: Array[(DecisionTreeModel, (String, Int, Int), MulticlassMetrics)]
    = DTreeUtil.multiParamDts(dataTrain, dataCV, 7, Map[Int,Int]() , Array("gini", "entropy"), Array(10, 20, 30), Array(40, 300) )
    /*
    evaluations.foreach(println)
    ((entropy,20,300),0.9119046392195256))
    ((gini   ,20,300),0.9058758867075454)
    ((entropy,20,10 ),0.8968585218391989)
    ((gini   ,20,10 ),0.89050342659865)
    ((gini   ,1 ,10 ),0.6330018378248399)
    ((gini   ,1 ,300),0.6323319764346198)
    ((entropy,1 ,300),0.48406932206592124)
    ((entropy,1 ,10 ),0.48406932206592124)

    val modelOpt = DecisionTree.trainClassifier( dataTrain.union(dataCV), 7, Map[Int,Int](), "entropy", 20, 300 )
    val model = evaluations.head._1
    val metrics = DTreeUtil.getMetrics(model, dataTest)
    metrics.precision : Double = 0.9161946933031271
    metrics.confusionMatrix: Matrix =
    19253.0  1781.0   1.0     0.0    13.0   1.0     80.0
    1913.0   26274.0  68.0    0.0    84.0   60.0    15.0
    0.0      73.0     3397.0  29.0   14.0   114.0   0.0
    0.0      0.0      29.0    230.0  0.0    12.0    0.0
    21.0     202.0    5.0     0.0    806.0  5.0     0.0
    4.0      78.0     123.0   13.0   3.0    1514.0  0.0
    132.0    10.0     0.0     0.0    0.0    0.0     1909.0
    val metricsFit = DTreeUtil.getMetrics(model, dataTrain.union(dataCV))
    metricsFit.precision: Double = 0.9520206754331931
     */
    evaluations
  }

  def catMultiParamDt(sc:SparkContext,  args: Array[String]) = {
    val Array(inpath) = args
    val dataLabeled = DTreeUtil.covtype2CatFeature(sc, inpath)
    val Array(dataTrain, dataCV, dataTest) = dataLabeled.randomSplit(Array(0.8, 0.1, 0.1))
    dataTrain.cache();dataCV.cache();dataTest.cache()
    val evaluations: Array[(DecisionTreeModel, (String, Int, Int), MulticlassMetrics)]
    = DTreeUtil.multiParamDts(dataTrain, dataCV, 7, Map(10 -> 4, 11 -> 40), Array("gini", "entropy"), Array(10, 20, 30), Array(40, 300) )
    /*
    evaluations.foreach(println)
((entropy,30,300),0.9446513552658804)
((gini,30,300),0.9391509759293745)
((entropy,30,40),0.9389268225394855)
((gini,30,40),0.9355817642596042)
((gini,20,300),0.9295123801641493)
((entropy,20,300),0.9285985240361404)
((entropy,20,40),0.9260121387681909)
((gini,20,40),0.9227877784674805)
((gini,10,300),0.792554658941996)
((entropy,10,300),0.7822263604386509)
((entropy,10,40),0.7815539002689841)
((gini,10,40),0.7813297468790951)

    val model = DecisionTree.trainClassifier( dataTrain.union(dataCV), 7, Map(10 -> 4, 11 -> 40), "entropy", 30, 300 )
    val model = evaluations.head._1
    val metrics = DTreeUtil.getMetrics(model, dataTest)
     */
    val model = evaluations.head._1

    evaluations
  }

  /*
fMeasure=0.9261723940869062, precision=0.9261723940869062, recall=0.9261723940869062
   */
  def catMultiParamDtCv(sc:SparkContext,  args: Array[String]) = {
    val Array(inpath) = args
    val numClasses = 7
    val catInfo = Map(10 -> 4, 11 -> 40)
    val maxBins = Array(40, 300)
    val maxDepth = Array(10, 20, 30)
    val impurities = Array(0,1)
    val numFolds = 3
    // load persisted training / testing dataset
    /*
    val labPnts: RDD[LabeledPoint] = DTreeUtil.covtype2CatFeature(sc, inpath)
    val Array(dataTrain, dataTest) = labPnts.randomSplit(Array(0.5, 0.5))
    */
    val dataTrain = sc.textFile("file:///home/leoricklin/dataset/covtype/data.Train").
      map { line =>
      val tokens = line.split(',').map(_.toDouble)
      LabeledPoint(tokens.last, Vectors.dense(tokens.init))
    }
    val dataTest = sc.textFile("file:///home/leoricklin/dataset/covtype/data.Test").
      map { line =>
      val tokens = line.split(',').map(_.toDouble)
      LabeledPoint(tokens.last, Vectors.dense(tokens.init))
    }
    // evaluating hyper-parameters
    val evaluations: Array[(Array[Int], Array[(DecisionTreeModel, Array[Double])])] = DTreeUtil.
      multiParamDtCvs( dataTrain, numClasses, catInfo,
        maxBins, maxDepth, impurities, numFolds)
    // print out all models & metrics
    /*
    evaluations.foreach{ case (params, modelMetrics) =>
        val metrics  = modelMetrics.map{ case (model, metrics) =>
            metrics.map{ metric => f"${metric}%5.5f" }.mkString("(",",",")") }.
          mkString(";")
        val avgs = modelMetrics.map { case (model, metrics) => metrics }.
          reduce( (ary1, ary2) =>
            Array( ary1(0)+ary2(0) , ary1(1)+ary2(1) , ary1(2)+ary2(2)) ).
          map{ metric => f"${metric / numFolds.toDouble}%5.5f"}.
          mkString("(",",",")")
        println(f"params:[${params.mkString(",")}] , metrics:\n ${metrics}}\n avg:${avgs}") }

params:[40,10,0] , metrics:
 (0.78516,0.78516,0.78516);(0.78207,0.78207,0.78207);(0.78703,0.78703,0.78703)}
 avg:(0.78475,0.78475,0.78475)
params:[300,10,0] , metrics:
 (0.79031,0.79031,0.79031);(0.78521,0.78521,0.78521);(0.79774,0.79774,0.79774)}
 avg:(0.79109,0.79109,0.79109)
params:[40,20,0] , metrics:
 (0.89860,0.89860,0.89860);(0.89793,0.89793,0.89793);(0.89933,0.89933,0.89933)}
 avg:(0.89862,0.89862,0.89862)
params:[300,20,0] , metrics:
 (0.89514,0.89514,0.89514);(0.89738,0.89738,0.89738);(0.89566,0.89566,0.89566)}
 avg:(0.89606,0.89606,0.89606)
params:[40,30,0] , metrics:
 (0.90486,0.90486,0.90486);(0.90238,0.90238,0.90238);(0.90553,0.90553,0.90553)}
 avg:(0.90426,0.90426,0.90426)
params:[300,30,0] , metrics:
 (0.90194,0.90194,0.90194);(0.90291,0.90291,0.90291);(0.90411,0.90411,0.90411)}
 avg:(0.90298,0.90298,0.90298)
params:[40,10,1] , metrics:
 (0.77460,0.77460,0.77460);(0.77938,0.77938,0.77938);(0.77386,0.77386,0.77386)}
 avg:(0.77595,0.77595,0.77595)
params:[300,10,1] , metrics:
 (0.77765,0.77765,0.77765);(0.77445,0.77445,0.77445);(0.77621,0.77621,0.77621)}
 avg:(0.77610,0.77610,0.77610)
params:[40,20,1] , metrics:
 (0.89912,0.89912,0.89912);(0.90031,0.90031,0.90031);(0.89936,0.89936,0.89936)}
 avg:(0.89960,0.89960,0.89960)
params:[300,20,1] , metrics:
 (0.89853,0.89853,0.89853);(0.90004,0.90004,0.90004);(0.89658,0.89658,0.89658)}
 avg:(0.89838,0.89838,0.89838)
params:[40,30,1] , metrics:
 (0.90912,0.90912,0.90912);(0.90806,0.90806,0.90806);(0.90607,0.90607,0.90607)}
 avg:(0.90775,0.90775,0.90775)
params:[300,30,1] , metrics:
 (0.90971,0.90971,0.90971);(0.91125,0.91125,0.91125);(0.91024,0.91024,0.91024)}
 avg:(0.91040,0.91040,0.91040)
     */
    val modelParams: (Array[Int], Array[(DecisionTreeModel, Array[Double])]) = evaluations.sortBy{ case (params, modelMetrics) =>
        modelMetrics.map{ case (model, metrics) => metrics(1)}.max }.
      last
    /*
: (Array[Int], Array[(org.apache.spark.mllib.tree.model.DecisionTreeModel, Array[Double])])
 = (Array(300, 30, 1),Array(
 (DecisionTreeModel classifier of depth 30 with 25931 nodes,Array(0.909705216825269,  0.909705216825269,  0.909705216825269)),
 (DecisionTreeModel classifier of depth 30 with 25477 nodes,Array(0.9112546752805168, 0.9112546752805168, 0.9112546752805168)),
 (DecisionTreeModel classifier of depth 30 with 26479 nodes,Array(0.91023970548696,   0.91023970548696,   0.91023970548696))))
     */
    // training model
    dataTrain.cache()
    val bins = modelParams._1(0)
    val depth = modelParams._1(1)
    val impurity = modelParams._1(2) match {
      case 0 => "gini"
      case _ => "entropy"
    }
    val modelTrain = DecisionTree.trainClassifier(dataTrain, numClasses, catInfo,
      impurity, depth, bins)
    dataTrain.unpersist()
    // testing model
    dataTest.cache()
    val predictLabels = modelTrain.
      predict( dataTest.map(_.features) ).
      zip(dataTest.map(_.label))
    dataTest.unpersist()
    val metric = new MulticlassMetrics(predictLabels)
    println(f"fMeasure=${metric.fMeasure}, precision=${metric.precision}, recall=${metric.recall}")
    /*
fMeasure=0.9261723940869062, precision=0.9261723940869062, recall=0.9261723940869062
     */
    val t1 = LabeledPoint(1.0, Vectors.dense(Array(0.1D)))
    val t2 = Array(1.0D,2.0D,3.0D)
    val t3 = t2.map(it => Vectors.dense(Array(it)))
  }

  def catSimpleRf(sc:SparkContext, args: Array[String]) = {
    val Array(inpath) = args
    val dataLabeled = DTreeUtil.covtype2CatFeature(sc, inpath)
    val Array(dataTrain, dataCV, dataTest) = dataLabeled.randomSplit(Array(0.8, 0.1, 0.1))
    dataTrain.cache();dataCV.cache();dataTest.cache()
    //
    val model: RandomForestModel = RandomForest.trainClassifier(dataTrain, 7, Map(10 -> 4, 11 -> 40), 20, "auto", "entropy", 30, 300)
    /*
15/09/09 11:09:13 INFO RandomForest:   init: 7.010611743
  total: 929.750164607
  findSplitsBins: 0.736501206
  findBestSplits: 921.283533547
  chooseSplits: 920.719117937
model: org.apache.spark.mllib.tree.model.RandomForestModel = TreeEnsembleModel classifier with 20 trees
     */
    val predictionsAndLabels = dataTest.map(example => (model.predict(example.features), example.label) )
    val metrics = new MulticlassMetrics(predictionsAndLabels)
    /*
    metrics.precision : Double = 0.9630068932322555
    metrics.confusionMatrix: Matrix =
20315.0  967.0    0.0     0.0    8.0    6.0     31.0
466.0    27758.0  52.0    0.0    51.0   37.0    8.0
0.0      47.0     3392.0  14.0   2.0    67.0    0.0
0.0      0.0      28.0    212.0  0.0    5.0     0.0
2.0      109.0    11.0    0.0    834.0  2.0     0.0
1.0      35.0     109.0   6.0    4.0    1529.0  0.0
75.0     9.0      0.0     0.0    0.0    0.0     1981.0
     */
    val model2 = TimeEvaluation.time( RandomForest.trainClassifier(dataTrain.union(dataCV), 7, Map(10 -> 4, 11 -> 40), 20, "auto", "entropy", 30, 300) )
    /*
[local mode, 16 cores/32GB]
time: 1018484.382225ms
model2: org.apache.spark.mllib.tree.model.RandomForestModel = TreeEnsembleModel classifier with 20 trees
[cluster mode, 12 cores/48GB * 2 nodes]
time: 1088371.736012ms
model2: org.apache.spark.mllib.tree.model.RandomForestModel = TreeEnsembleModel classifier with 20 trees

     */
    val predictionsAndLabels2 = dataTest.map(example => (model.predict(example.features), example.label) )
    val metrics2 = new MulticlassMetrics(predictionsAndLabels)
    /*
    metrics.precision = 0.9630068932322555
    metrics.confusionMatrix
20315.0  967.0    0.0     0.0    8.0    6.0     31.0
466.0    27758.0  52.0    0.0    51.0   37.0    8.0
0.0      47.0     3392.0  14.0   2.0    67.0    0.0
0.0      0.0      28.0    212.0  0.0    5.0     0.0
2.0      109.0    11.0    0.0    834.0  2.0     0.0
1.0      35.0     109.0   6.0    4.0    1529.0  0.0
75.0     9.0      0.0     0.0    0.0    0.0     1981.0
     */

  }

  /*
fMeasure=0.9518270137774809, precision=0.9518270137774809, recall=0.9518270137774809, numTrees = 16
fMeasure=0.9561440955128514, precision=0.9561440955128514, recall=0.9561440955128514, numTrees = 32
fMeasure=0.9510558604212426, precision=0.9510558604212426, recall=0.9510558604212426, numTrees = 32, Stacking with meta-features
fMeasure=0.9510937295592721, precision=0.9510937295592721, recall=0.9510937295592721, numTrees = 32, Stacking with all features
fMeasure=0.9568360679441189, precision=0.9568360679441189, recall=0.9568360679441189, numTrees = 32, Retrain tier-1 model and stacking with all features
   */

  def catMultiParamRfCv(sc:SparkContext,  args: Array[String]) = {
    val Array(inpath) = args
    val numClasses = 7
    val catInfo = Map(10 -> 4, 11 -> 40)
    val maxBins = Array(40, 300)
    val maxDepth = Array(10, 30)
    val impurities = Array(0, 1)
    val maxTrees = Array(16, 32)
    val numFolds = 3
    val seed = 1
    // load persisted training / testing dataset
    /*
    val labPnts: RDD[LabeledPoint] = DTreeUtil.covtype2CatFeature(sc, inpath)
    val Array(dataTrain, dataTest) = labPnts.randomSplit(Array(0.5, 0.5))
     */
    val dataTrain = sc.textFile("file:///home/leoricklin/dataset/covtype/data.Train").
      map { line =>
      val tokens = line.split(',').map(_.toDouble)
      LabeledPoint(tokens.last, Vectors.dense(tokens.init))
    }
    val dataTest = sc.textFile("file:///home/leoricklin/dataset/covtype/data.Test").
      map { line =>
      val tokens = line.split(',').map(_.toDouble)
      LabeledPoint(tokens.last, Vectors.dense(tokens.init))
    }
    /*
    dataTrain.count = 290538
    dataTest.count = 290474
*/
    // persist training / testing dataset
    /*
    dataTrain.map{ lp =>
        (lp.features.toArray.map{_.toString} :+ lp.label.toString).mkString(",") }.
      saveAsTextFile("file:///home/leoricklin/dataset/covtype/data.Train")
    dataTest.map{ lp =>
      (lp.features.toArray.map{_.toString} :+ lp.label.toString).mkString(",") }.
      saveAsTextFile("file:///home/leoricklin/dataset/covtype/data.Test")
*/

    // numTrees = 16
    val model1 = {
      val maxTrees = Array(16)
      // evaluating hyper-parameters
      val evaluations: Array[(Array[Int], Array[(RandomForestModel, Array[Double])])] = TimeEvaluation.time(DTreeUtil.
        multiParamRfCvs(dataTrain, numClasses, catInfo,
          maxBins, maxDepth, impurities, maxTrees, numFolds))
      /*
time: 3,087,401.17096ms
       */
      // get best parameters by precision
      val modelParams: (Array[Int], Array[(RandomForestModel, Array[Double])]) = evaluations.
        sortBy { case (params, modelMetrics) =>
          modelMetrics.map { case (model, metrics) => metrics(1)}.max }.
        last
      /*
(Array(300, 30, 1, 16),Array(
(TreeEnsembleModel classifier with 16 trees, Array(0.9408548245387358, 0.9408548245387358, 0.9408548245387358)),
(TreeEnsembleModel classifier with 16 trees ,Array(0.9418153367854312, 0.9418153367854312, 0.9418153367854312)),
(TreeEnsembleModel classifier with 16 trees ,Array(0.9419992964595361, 0.9419992964595361, 0.9419992964595361))))
       */
      // training best model
      dataTrain.cache()
      val Array(bins, depth, numImpurity, trees) = modelParams._1
      val impurity = numImpurity match {
        case 0 => "gini"
        case _ => "entropy"
      }
      val modelTrain = TimeEvaluation.time(RandomForest.trainClassifier(dataTrain, numClasses, catInfo,
        trees, "auto", impurity, depth, bins, seed))
      /*
time: 408,607.696624ms
       */
      dataTrain.unpersist()
      // testing best model
      dataTest.cache()
      val predictLabels = modelTrain.
        predict(dataTest.map(_.features)).
        zip(dataTest.map(_.label))
      dataTest.unpersist()
      val metric = new MulticlassMetrics(predictLabels)
      println(f"fMeasure=${metric.fMeasure}, precision=${metric.precision}, recall=${metric.recall}")
      /*
fMeasure=0.9518270137774809, precision=0.9518270137774809, recall=0.9518270137774809
      */
    }

    // numTrees = 32
    val model2 = {
      val maxTrees = Array(32)
      // evaluating hyper-parameters
      val evaluations: Array[(Array[Int], Array[(RandomForestModel, Array[Double])])] = TimeEvaluation.time(DTreeUtil.
        multiParamRfCvs(dataTrain, numClasses, catInfo,
          maxBins, maxDepth, impurities, maxTrees, numFolds))
      /*
time: 7,766,210.85376ms
       */
      // print out parameters
      evaluations.foreach { case (params, modelMetrics) =>
        val metrics = modelMetrics.map { case (model, metrics) =>
            metrics.map { metric => f"${metric}%5.5f"}.mkString("(", ",", ")") }.
          mkString(";")
        val avgs = modelMetrics.map { case (model, metrics) => metrics}.
          reduce{ (ary1, ary2) =>
            Array(ary1(0) + ary2(0), ary1(1) + ary2(1), ary1(2) + ary2(2)) }.
          map { metric => f"${metric / numFolds.toDouble}%5.5f" }.
          mkString("(", ",", ")")
        println( f"params:[${params.mkString(",")}] , metrics:\n ${metrics}}\n avg:${avgs}" )
      }
      /*
params:[40,10,0,32] , metrics:  (time: 27,802.309679ms, 26807.389685ms, 25919.269708ms)
 (0.78503,0.78503,0.78503);(0.78228,0.78228,0.78228);(0.78414,0.78414,0.78414)}
 avg:(0.78382,0.78382,0.78382)
params:[40,10,1,32] , metrics:  (time: 27,121.79256ms, 25801.705976ms, 27319.529973ms)
 (0.78167,0.78167,0.78167);(0.78528,0.78528,0.78528);(0.78191,0.78191,0.78191)}
 avg:(0.78295,0.78295,0.78295)
params:[40,30,0,32] , metrics:  (time: 394,901.751334ms, 394911.169032ms, 390580.594618ms)
 (0.93895,0.93895,0.93895);(0.93905,0.93905,0.93905);(0.94027,0.94027,0.94027)}
 avg:(0.93942,0.93942,0.93942)
params:[40,30,1,32] , metrics:  (time: 361,958.71377ms, 365,133.022628ms, 366,542.977985ms )
 (0.94182,0.94182,0.94182);(0.94204,0.94204,0.94204);(0.94365,0.94365,0.94365)}
 avg:(0.94250,0.94250,0.94250)
params:[300,10,0,32] , metrics: (time: 38,206.980747ms, 38304.031689ms, 38074.857951ms)
 (0.78428,0.78428,0.78428);(0.78437,0.78437,0.78437);(0.78611,0.78611,0.78611)}
 avg:(0.78492,0.78492,0.78492)
params:[300,10,1,32] , metrics: (time: 38,958.499563ms, 39450.816363ms, 40106.944404ms)
 (0.78579,0.78579,0.78579);(0.78680,0.78680,0.78680);(0.78795,0.78795,0.78795)}
 avg:(0.78685,0.78685,0.78685)
params:[300,30,0,32] , metrics: (time: 831,008.866137ms, 845169.886808ms, 847757.742271ms)
 (0.94152,0.94152,0.94152);(0.94183,0.94183,0.94183);(0.94368,0.94368,0.94368)}
 avg:(0.94234,0.94234,0.94234)
params:[300,30,1,32] , metrics: (time: 740,711.823982ms, 731,518.798112ms , 743,741.309108ms)
 (0.94531,0.94531,0.94531);(0.94642,0.94642,0.94642);(0.94702,0.94702,0.94702)}
 avg:(0.94625,0.94625,0.94625)
       */
      // get best parameters by precision
      val modelParams = evaluations.
        sortBy { case (params, modelMetrics) =>
          modelMetrics.map { case (model, metrics) => metrics(1)}.max }.
        last
      /*
(Array(300, 30, 1, 32),Array(
(TreeEnsembleModel classifier with 32 trees ,Array(0.9453098351335987,  0.9453098351335987, 0.9453098351335987)),
(TreeEnsembleModel classifier with 32 trees ,Array(0.946416990086371,   0.946416990086371,  0.946416990086371)),
(TreeEnsembleModel classifier with 32 trees ,Array(0.947017195356633,   0.947017195356633,  0.947017195356633))))
       */
      // training best model
      dataTrain.cache()
      val Array(bins, depth, numImpurity, trees) = modelParams._1
      val impurity = numImpurity match {
        case 0 => "gini"
        case _ => "entropy"
      }
      val modelTrain = TimeEvaluation.time(RandomForest.trainClassifier(dataTrain, numClasses, catInfo,
        trees, "auto", impurity, depth, bins, seed))
      /*
time: 1,170,769.58043ms
       */
      dataTrain.unpersist()
      // testing best model
      dataTest.cache()
      val predictLabels = modelTrain.
        predict(dataTest.map(_.features)).
        zip(dataTest.map(_.label))
      dataTest.unpersist()
      val metric = new MulticlassMetrics(predictLabels)
      println(f"fMeasure=${metric.fMeasure}, precision=${metric.precision}, recall=${metric.recall}")
      /*
fMeasure=0.9561440955128514, precision=0.9561440955128514, recall=0.9561440955128514
       */
    }

    // numTrees = 32, Stacking with meta-feature
    val model3 = {
      val maxTrees = Array(32)
      // evaluating hyper-parameters
      val evaluations: Array[(Array[Int], Array[(RandomForestModel, Array[Double])])] = TimeEvaluation.time(DTreeUtil.
        multiParamRfCvs(dataTrain, numClasses, catInfo, maxBins, maxDepth, impurities, maxTrees, numFolds))
      // get best parameters by precision
      val modelParams = evaluations.
        sortBy { case (params, modelMetrics) =>
        modelMetrics.map { case (model, metrics) => metrics(1)}.max }.
        last
      // get best models by precision
      val models: Array[(RandomForestModel, Array[Double])] = evaluations.
        flatMap{ case (params, modelMetrics) => modelMetrics }.
        sortBy { case (model, metrics) => metrics(1) }.
        takeRight(3)
      /*
Array(
(TreeEnsembleModel classifier with 32 trees, Array(0.9453098351335987,  0.9453098351335987, 0.9453098351335987)),
(TreeEnsembleModel classifier with 32 trees, Array(0.946416990086371,   0.946416990086371,  0.946416990086371)),
(TreeEnsembleModel classifier with 32 trees, Array(0.947017195356633,   0.947017195356633,  0.947017195356633)))
       */
      // preparing meta features
      dataTrain.cache()
      val trainPredictions: Array[RDD[Double]] = models.map{ case (model, metrics) => model }.
        map{ model => model.predict{ dataTrain.map{ _.features} } }
      val dataTrainMeta: RDD[LabeledPoint] = dataTrain.
        zip{ trainPredictions(0) }.
        map{ case (lp, prediction) => (lp, Array(prediction)) }.
        zip{ trainPredictions(1) }.
        map{ case ((lp, features), prediction) => (lp, features :+ prediction)}.
        zip{ trainPredictions(2) }.
        map{ case ((lp, features), prediction) => (lp, features :+ prediction)}.
        map{ case (lp, features) => LabeledPoint(lp.label, Vectors.dense(features)) }
      /*
dataTrainPred.first: LabeledPoint = (0.0,[0.0,0.0,0.0])
       */
      dataTrain.unpersist()
      // training tier-2 model
      dataTrainMeta.cache()
      val Array(bins, depth, numImpurity, trees) = modelParams._1
      val impurity = numImpurity match {
        case 0 => "gini"
        case _ => "entropy"
      }
      val catInfoNew = Map(0 -> 7, 1 -> 7, 2 -> 7)
      val modelStack = TimeEvaluation.time(RandomForest.trainClassifier(dataTrainMeta, numClasses, catInfoNew,
        trees, "auto", impurity, depth, bins, seed))
      /*
time: 389,375.69119ms
       */
      dataTrainMeta.unpersist()
      // testing tier-2 model
      dataTest.cache()
      /* the prediction get much worse if don't transform testing data into meta features
      val predictLabels = modelStack.
        predict(dataTest.map(_.features)).
        zip(dataTest.map(_.label))
      val metric = new MulticlassMetrics(predictLabels)
      println(f"fMeasure=${metric.fMeasure}, precision=${metric.precision}, recall=${metric.recall}")
fMeasure=0.03468813043508197, precision=0.03468813043508197, recall=0.03468813043508197
       */
      // preparing meta features
      val testPredictions = models.map{ case (model, metrics) => model }.
        map{ model => model.predict{ dataTest.map{ _.features} } }
      val dataTestMeta: RDD[LabeledPoint] = dataTest.
        zip{ testPredictions(0) }.
        map{ case (lp, prediction) => (lp, Array(prediction)) }.
        zip{ testPredictions(1) }.
        map{ case ((lp, features), prediction) => (lp, features :+ prediction)}.
        zip{ testPredictions(2) }.
        map{ case ((lp, features), prediction) => (lp, features :+ prediction)}.
        map{ case (lp, features) => LabeledPoint(lp.label, Vectors.dense(features)) }
      val predictLabels = modelStack.
        predict(dataTestMeta.map(_.features)).
        zip(dataTestMeta.map(_.label))
      dataTest.unpersist()
      val metric = new MulticlassMetrics(predictLabels)
      println(f"fMeasure=${metric.fMeasure}, precision=${metric.precision}, recall=${metric.recall}")
      /*
fMeasure=0.9510558604212426, precision=0.9510558604212426, recall=0.9510558604212426
       */
    }

    // numTrees = 32. Stacking with original features and meta-feature
    val model4 = {
      val maxTrees = Array(32)
      // evaluating hyper-parameters
      val evaluations: Array[(Array[Int], Array[(RandomForestModel, Array[Double])])] = TimeEvaluation.time(DTreeUtil.
        multiParamRfCvs(dataTrain, numClasses, catInfo,
          maxBins, maxDepth, impurities, maxTrees, numFolds))
      // get best parameters by precision
      val modelParams = evaluations.
        sortBy { case (params, modelMetrics) =>
        modelMetrics.map { case (model, metrics) => metrics(1)}.max }.
        last
      // get best models by precision
      val models: Array[(RandomForestModel, Array[Double])] = evaluations.
        flatMap{ case (params, modelMetrics) => modelMetrics }.
        sortBy { case (model, metrics) => metrics(1) }.
        takeRight(3)
      /*
models: Array[(RandomForestModel, Array[Double])] = Array(
(TreeEnsembleModel classifier with 32 trees ,Array(0.9453098351335987, 0.9453098351335987, 0.9453098351335987)),
(TreeEnsembleModel classifier with 32 trees ,Array(0.946416990086371, 0.946416990086371, 0.946416990086371)),
(TreeEnsembleModel classifier with 32 trees ,Array(0.947017195356633, 0.947017195356633, 0.947017195356633)))
       */
      // preparing meta features
      dataTrain.cache()
      val trainPredictions: Array[RDD[Double]] = models.map{ case (model, metrics) =>
          model.predict{ dataTrain.map{ _.features} } }
      val dataTrainMeta: RDD[LabeledPoint] = dataTrain.
        zip{ trainPredictions(0) }.
        map{ case (lp, prediction) => (lp, Array(prediction)) }.
        zip{ trainPredictions(1) }.
        map{ case ((lp, features), prediction) => (lp, features :+ prediction)}.
        zip{ trainPredictions(2) }.
        map{ case ((lp, features), prediction) => (lp, features :+ prediction)}.
        map{ case (lp, features) => LabeledPoint(lp.label, Vectors.dense(lp.features.toArray ++: features)) }
      /*
dataTrainMeta.first: LabeledPoint = (0.0,[3266.0,302.0,15.0,516.0,84.0,666.0,178.0,232.0,194.0,247.0,1.0,23.0,0.0,0.0,0.0])
       */
      dataTrain.unpersist()
      // training tier-2 model
      dataTrainMeta.cache()
      val Array(bins, depth, numImpurity, trees) = modelParams._1
      val impurity = numImpurity match {
        case 0 => "gini"
        case _ => "entropy" }
      val catInfoNew = Map(10 -> 4, 11 -> 40, 12 -> 7, 13 -> 7, 14 -> 7)
      val modelStack = TimeEvaluation.time(RandomForest.trainClassifier(dataTrainMeta, numClasses, catInfoNew,
        trees, "auto", impurity, depth, bins, seed))
      /*
time: 757,623.002274ms
       */
      dataTrainMeta.unpersist()
      // testing tier-2 model
      dataTest.cache()
      // preparing meta features
      val testPredictions = models.map{ case (model, metrics) => model }.
        map{ model => model.predict{ dataTest.map{ _.features} } }
      val dataTestMeta: RDD[LabeledPoint] = dataTest.
        zip{ testPredictions(0) }.
        map{ case (lp, prediction) => (lp, Array(prediction)) }.
        zip{ testPredictions(1) }.
        map{ case ((lp, features), prediction) => (lp, features :+ prediction)}.
        zip{ testPredictions(2) }.
        map{ case ((lp, features), prediction) => (lp, features :+ prediction)}.
        map{ case (lp, features) => LabeledPoint(lp.label, Vectors.dense(lp.features.toArray ++: features)) }
      /*
dataTestMeta.first: LabeledPoint = (1.0,[3250.0,103.0,18.0,300.0,55.0,150.0,247.0,214.0,89.0,603.0,1.0,31.0,1.0,1.0,1.0])
       */
      val predictLabels = modelStack.predict(dataTestMeta.map(_.features)).
        zip(dataTestMeta.map(_.label))
      dataTest.unpersist()
      val metric = new MulticlassMetrics(predictLabels)
      println(f"fMeasure=${metric.fMeasure}, precision=${metric.precision}, recall=${metric.recall}")
      /*
fMeasure=0.9510937295592721, precision=0.9510937295592721, recall=0.9510937295592721
       */
    }

    // Retrain tier-1 model and stacking with all features
    val model5 = {
      val maxTrees = Array(32)
      // evaluating hyper-parameters
      val evaluations: Array[(Array[Int], Array[(RandomForestModel, Array[Double])])] = TimeEvaluation.time(DTreeUtil.
        multiParamRfCvs(dataTrain, numClasses, catInfo,
          maxBins, maxDepth, impurities, maxTrees, numFolds))
      // get best parameters by precision
      val modelParams: Array[Array[Int]] = evaluations.
        sortBy { case (params, modelMetrics) =>
          modelMetrics.map { case (model, metrics) => metrics(1)}.max }.
        takeRight(3).
        map{ case (params, modelMetrics) => params }
      // training tier-1 models
      dataTrain.cache()
      val models: Array[RandomForestModel] = for { params <- modelParams } yield {
        val Array(bins, depth, numImpurity, trees) = params
        val impurity = numImpurity match {
          case 0 => "gini"
          case _ => "entropy" }
        val modelTrain = TimeEvaluation.time( RandomForest.trainClassifier(dataTrain, numClasses, catInfo,
          trees, "auto", impurity, depth, bins, seed))
        modelTrain
      }
      // training tier-2 model
      val trainAndMeta: RDD[LabeledPoint] = DTreeUtil.appendPredictions(dataTrain, models)
      /*
trainAndMeta.first: LabeledPoint = (0.0,[3266.0,302.0,15.0,516.0,84.0,666.0,178.0,232.0,194.0,247.0,1.0,23.0,0.0,0.0,0.0])
       */
      trainAndMeta.cache()
      val Array(bins, depth, numImpurity, trees) = modelParams.last
      val impurity = numImpurity match {
        case 0 => "gini"
        case _ => "entropy" }
      val catInfoNew = Map(10 -> 4, 11 -> 40, 12 -> 7, 13 -> 7, 14 -> 7)
      val modelStack = TimeEvaluation.time(RandomForest.trainClassifier(trainAndMeta, numClasses, catInfoNew,
        trees, "auto", impurity, depth, bins, seed))
      trainAndMeta.unpersist()
      // testing tier-2 model
      val testAndMeta = DTreeUtil.appendPredictions(dataTest, models)
      /*
testAndMeta.first: LabeledPoint = (1.0,[3250.0,103.0,18.0,300.0,55.0,150.0,247.0,214.0,89.0,603.0,1.0,31.0,1.0,1.0,1.0])
       */
      testAndMeta.cache()
      val predictLabels = modelStack.predict( testAndMeta.map(_.features) ).
        zip(testAndMeta.map(_.label))
      testAndMeta.unpersist()
      val metric = new MulticlassMetrics(predictLabels)
      println(f"fMeasure=${metric.fMeasure}, precision=${metric.precision}, recall=${metric.recall}")
      /*
fMeasure=0.9568360679441189, precision=0.9568360679441189, recall=0.9568360679441189
       */
      val rootNodes = modelStack.trees.map{_.topNode}
      /*
       modelStack.trees(0).numNodes = 105
       */
      val nodeRoot = modelStack.trees(0).topNode
      /*
id = 1, isLeaf = false, predict = 1.0 (prob = 0.48701564732411823), impurity = 1.7360444430062776
, split = Some(Feature = 13, threshold = -1.7976931348623157E308, featureType = Categorical, categories = List(4.0, 1.0))
, stats = Some(gain = 0.9978037506669467, impurity = 1.7360444430062776, left impurity = 0.2091823431342555, right impurity = 1.2744025879760192)
       */
      var nodeParent = nodeRoot.leftNode.get
      /*
id = 2, isLeaf = false, predict = 1.0 (prob = 0.9675583614339163), impurity = 0.2091823431342555
, split = Some(Feature = 12, threshold = -1.7976931348623157E308, featureType = Categorical, categories = List(4.0))
, stats = Some(gain = 0.20452972192559485, impurity = 0.2091823431342555, left impurity = 0.0, right impurity = 0.004806955651150774)
       */
      var nodeLeft = if (!nodeParent.isLeaf) {nodeParent.leftNode.get} else {nodeParent}
      /*
id = 4, isLeaf = true, predict = 4.0 (prob = 1.0), impurity = 0.0, split = None, stats = None
       */
      var nodeRight = if (!nodeParent.isLeaf) {nodeParent.rightNode.get} else {nodeParent}
      /*
id = 5, isLeaf = false, predict = 1.0 (prob = 0.9996537273774442), impurity = 0.004806955651150774
, split = Some(Feature = 12, threshold = -1.7976931348623157E308, featureType = Categorical, categories = List(1.0))
, stats = Some(gain = 0.0023935099074746625, impurity = 0.004806955651150774, left impurity = 0.001829786518481671, right impurity = 1.1817135862269055)
       */
      nodeParent = nodeLeft  // id=4
      nodeParent = nodeRight // id=5
      nodeLeft = if (!nodeParent.isLeaf) {nodeParent.leftNode.get} else {nodeParent}
      /*
id = 10, isLeaf = false, predict = 1.0 (prob = 0.9998868754286361), impurity = 0.001829786518481671
 , split = Some(Feature = 14, threshold = -1.7976931348623157E308, featureType = Categorical, categories = List(2.0, 1.0))
 , stats = Some(gain = 7.38708585706019E-4, impurity = 0.001829786518481671, left impurity = 9.337396292840691E-4, right impurity = 1.8553885422075338)
       */
      nodeRight = if (!nodeParent.isLeaf) {nodeParent.rightNode.get} else {nodeParent}
      /*
id = 11, isLeaf = false, predict = 1.0 (prob = 0.5285714285714286), impurity = 1.1817135862269055
 , split = Some(Feature = 14, threshold = -1.7976931348623157E308, featureType = Categorical, categories = List(0.0))
 , stats = Some(gain = 0.36846102674432113, impurity = 1.1817135862269055, left impurity = 0.5916727785823275, right impurity = 0.960972413416089)
       */
      nodeParent = nodeLeft  // id=10
      nodeLeft = if (!nodeParent.isLeaf) {nodeParent.leftNode.get} else {nodeParent}
      /*
id = 20, isLeaf = false, predict = 1.0 (prob = 0.9999434329149726), impurity = 9.337396292840691E-4
, split = Some(Feature = 4, threshold = 96.0, featureType = Continuous, categories = List())
, stats = Some(gain = 6.754531105176417E-5, impurity = 9.337396292840691E-4, left impurity = 6.596931313782946E-4, right impurity = 0.002113734971100895)
       */
      nodeRight = if (!nodeParent.isLeaf) {nodeParent.rightNode.get} else {nodeParent}
      /*
id = 21, isLeaf = false, predict = 0.0 (prob = 0.3333333333333333), impurity = 1.8553885422075338
, split = Some(Feature = 8, threshold = 123.0, featureType = Continuous, categories = List())
, stats = Some(gain = 0.9798687566511526, impurity = 1.8553885422075338, left impurity = 0.7219280948873623, right impurity = 0.9852281360342516)
       */
      nodeParent = nodeLeft // id=20
      nodeLeft = if (!nodeParent.isLeaf) {nodeParent.leftNode.get} else {nodeParent}
      DTUtil.printNode(nodeLeft) // id=40, feature=0, impurity=0.00066, cond=3017.00000
      nodeRight = if (!nodeParent.isLeaf) {nodeParent.rightNode.get} else {nodeParent}
      DTUtil.printNode(nodeRight) // id=41, feature=4, impurity=0.00211, cond=97.00000

    }
  }

  def main(args: Array[String]) {
    // :load /media/sf_WORKSPACE.2L/spark-adv/src/main/scala/ch04/DTreeUtil.scala
    // :load /media/sf_WORKSPACE.2L/spark-util/src/main/scala/tw/com/chttl/spark/test/util/TimeEvaluation.scala
    // val args = Array("file:///media/sf_WORKSPACE.2L/dataset/AAS/covtype/covtype.data")
    //
    // :load /home/leoricklin/jar/DTreeUtil.scala
    // val args = Array("file:///home/leoricklin/dataset/covtype/covtype.data")
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)
    if (args.length != 1) {
      System.err.printf("Arguments:%s\n", args.mkString("[",",","]"))
      System.err.printf("Usage: %s <input_path>", appName)
      System.exit(1)
    }
    var result: Array[(DecisionTreeModel, (String, Int, Int), MulticlassMetrics)] =
      TimeEvaluation.time( oneHotMultiParamDt(sc, args) )
    println(s"impurity:${result.head._2._1}, depth:${result.head._2._2}, bins:${result.head._2._3}")
    /*
    time(dTSimple(sc,  args))               time: 13948.195189ms
    time(dTMultiHyperParam(sc,  args))      time: 662567.335426ms
    time( dTCatMultiHyperParam(sc,  args) ) time: 673124.942699ms
     impurity:entropy, depth:30, bins:300
     */
    var model = result.head._1
    val pw = new PrintWriter(new File("/home/leoricklin/dataset/covtype/catSimpleRfModel2.log" ))
    pw.write(model.toDebugString)
    pw.close
  }
}
