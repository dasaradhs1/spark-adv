package ch04

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.{IDF, IDFModel, HashingTF}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree.{RandomForest, DecisionTree}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import tw.com.chttl.spark.test.util._

/**
 * Created by leorick on 2015/8/13.
 */
object DTreeUtil extends Serializable {
  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

  def simpleDecisionTree(trainData: RDD[LabeledPoint], cvData: RDD[LabeledPoint]): Unit = {
    // Build a simple default DecisionTreeModel
    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), "gini", 4, 100)

    val metrics = getMetrics(model, cvData)

    println(metrics.confusionMatrix)
    println(metrics.precision)

    (0 until 7).map(
      category => (metrics.precision(category), metrics.recall(category))
    ).foreach(println)
  }

  def randomClassifier(trainData: RDD[LabeledPoint], cvData: RDD[LabeledPoint]): Unit = {
    val trainPriorProbabilities = classProbabilities(trainData).map(_._2)
    val cvPriorProbabilities = classProbabilities(cvData).map(_._2)
    val accuracy = trainPriorProbabilities.zip(cvPriorProbabilities).map {
      case (trainProb, cvProb) => trainProb * cvProb
    }.sum
    println(accuracy)
  }

  def classProbabilities(data: RDD[LabeledPoint]): Array[(Double,Double)] = {
    // Count (category,count) in data
    val countsByCategory = data.map(_.label).countByValue()
    // order counts by category and extract counts
    val counts = countsByCategory.toArray.sortBy(_._1)
    counts.map{ case (label, cnt) => (label, cnt.toDouble / counts.map(_._2).sum) }
  }

  def evaluate(
                trainData: RDD[LabeledPoint],
                cvData: RDD[LabeledPoint],
                testData: RDD[LabeledPoint]): Unit = {

    val evaluations =
      for (impurity <- Array("gini", "entropy");
           depth    <- Array(1, 20);
           bins     <- Array(10, 300))
      yield {
        val model = DecisionTree.trainClassifier(
          trainData, 7, Map[Int,Int](), impurity, depth, bins)
        val accuracy = getMetrics(model, cvData).precision
        ((impurity, depth, bins), accuracy)
      }

    evaluations.sortBy(_._2).reverse.foreach(println)

    val model = DecisionTree.trainClassifier(
      trainData.union(cvData), 7, Map[Int,Int](), "entropy", 20, 300)
    println(getMetrics(model, testData).precision)
    println(getMetrics(model, trainData.union(cvData)).precision)
  }

  def unencodeOneHot(rawData: RDD[String]): RDD[LabeledPoint] = {
    rawData.map { line =>
      val values = line.split(',').map(_.toDouble)
      // Which of 4 "wilderness" features is 1
      val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
      // Similarly for following 40 "soil" features
      val soil = values.slice(14, 54).indexOf(1.0).toDouble
      // Add derived features back to first 10
      val featureVector = Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }
  }

  def evaluateCategorical(rawData: RDD[String]): Unit = {

    val data = unencodeOneHot(rawData)

    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    val evaluations =
      for (impurity <- Array("gini", "entropy");
           depth    <- Array(10, 20, 30);
           bins     <- Array(40, 300))
      yield {
        // Specify value count for categorical features 10, 11
        val model = DecisionTree.trainClassifier(
          trainData, 7, Map(10 -> 4, 11 -> 40), impurity, depth, bins)
        val trainAccuracy = getMetrics(model, trainData).precision
        val cvAccuracy = getMetrics(model, cvData).precision
        // Return train and CV accuracy
        ((impurity, depth, bins), (trainAccuracy, cvAccuracy))
      }

    evaluations.sortBy(_._2._2).reverse.foreach(println)

    val model = DecisionTree.trainClassifier(
      trainData.union(cvData), 7, Map(10 -> 4, 11 -> 40), "entropy", 30, 300)
    println(getMetrics(model, testData).precision)

    trainData.unpersist()
    cvData.unpersist()
    testData.unpersist()
  }

  def evaluateForest(rawData: RDD[String]): Unit = {

    val data = unencodeOneHot(rawData)

    val Array(trainData, cvData) = data.randomSplit(Array(0.9, 0.1))
    trainData.cache()
    cvData.cache()

    val forest = RandomForest.trainClassifier(
      trainData, 7, Map(10 -> 4, 11 -> 40), 20, "auto", "entropy", 30, 300)

    val predictionsAndLabels = cvData.map(example =>
      (forest.predict(example.features), example.label)
    )
    println(new MulticlassMetrics(predictionsAndLabels).precision)

    val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
    val vector = Vectors.dense(input.split(',').map(_.toDouble))
    println(forest.predict(vector))
  }
  /*
  by leo
   */
  def covtype2OneHotFeature(sc:SparkContext, inpath:String) = {
    val dataRaw: RDD[String] = sc.textFile(inpath)
    val dataLabeled: RDD[LabeledPoint] = dataRaw.map { line =>
      val values:Array[Double] = line.split(',').map(_.toDouble)
      val featureVector = Vectors.dense(values.init) // (1)
      val label = values.last - 1  // (2)
      LabeledPoint(label, featureVector)
    }
    dataLabeled
  }

  def covtype2CatFeature(sc:SparkContext, inpath:String) = {
    val dataRaw: RDD[String] = sc.textFile(inpath)
    val dataLabeled: RDD[LabeledPoint] = dataRaw.map { line =>
      val values:Array[Double] = line.split(',').map(_.toDouble)
      val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
      val soil = values.slice(14, 54).indexOf(1.0).toDouble
      val featureVector = Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }
    dataLabeled
  }

  def multiParamDts(dataTrain:RDD[LabeledPoint], dataCV:RDD[LabeledPoint],
    numClasses: Int, catInfo: Map[Int, Int],
    impurities: Array[String], maxDepths: Array[Int], maxBins: Array[Int])
  : Array[(DecisionTreeModel, (String, Int, Int), MulticlassMetrics)] = {
    val evaluations = for {
      impurity <- impurities
      depth <- maxDepths
      bins <- maxBins
    } yield {
      val model: DecisionTreeModel = DecisionTree.trainClassifier(dataTrain, numClasses, catInfo, impurity, depth, bins)
      val metrics: MulticlassMetrics = this.getMetrics(model, dataCV)
      (model, (impurity, depth, bins), metrics)
    }
    evaluations.sortBy{ case (model, (impurity, depth, bins), metrics) => metrics.precision }.reverse
  }

  /**
   *
   * @param dataTrain
   * @param numClasses
   * @param catInfo
   * @param impurities: Array[Int], 0 -> "gini" , 1 -> "entropy"
   * @param maxDepths
   * @param maxBins
   * @param numFolds
   * @return Array[ (Array[Int], Array[(DecisionTreeModel, Array[Double])])] =
   *         Array( parameters = Array(bins, depth, impurity),
   *                modelMetrics = Array( ( model, metrics = Array(fMeasure, precision, recall) ) ) )
   */
  def multiParamDtCvs( dataTrain:RDD[LabeledPoint], numClasses: Int, catInfo: Map[Int, Int],
    maxBins: Array[Int], maxDepths: Array[Int], impurities: Array[Int],
    numFolds: Int = 3)
  : Array[(Array[Int], Array[(DecisionTreeModel, Array[Double])])] = {
    /*
    val impurity = "entropy"
    val depth = 30
    val bins = 300
    val numClasses = 7
    val numFolds = 3
    val catInfo = Map(10 -> 4, 11 -> 40)
     */
    val evaluations: Array[(Array[Int], Array[(DecisionTreeModel, Array[Double])])] = for {
      impurityNum <- impurities
      depth <- maxDepths
      bins <- maxBins
    } yield {
      val seed = 1
      val impurity = impurityNum match {
        case 0 => "gini"
        case _ => "entropy"
      }
      // refer CrossValidator.fit()
      val folds: Array[(RDD[LabeledPoint], RDD[LabeledPoint])] = MLUtils.kFold(dataTrain, numFolds, seed)
      // folds.size = 3
      val modelMetrics: Array[(DecisionTreeModel, Array[Double])] = folds.
        map { case (training, validation) =>
          training.cache;validation.cache()
          // training model
          val model: DecisionTreeModel = DecisionTree.trainClassifier(training, numClasses, catInfo, impurity, depth, bins)
          training.unpersist()
          // validatiing model
          val predictLabels: RDD[(Double, Double)] = model.predict( validation.map(_.features) ).
            zip( validation.map(_.label) )
          validation.unpersist()
          val metric = new MulticlassMetrics(predictLabels)
        ( model, Array(metric.fMeasure, metric.precision, metric.recall) )
      }
      ( Array(bins, depth, impurityNum), modelMetrics )
    }
    evaluations
  }

  /**
   *
   * @param dataTrain
   * @param numClasses
   * @param catInfo
   * @param maxBins
   * @param maxDepths
   * @param impurities
   * @param maxTrees
   * @param numFolds
   * @return Array[(Array[Int], Array[(RandomForestModel, Array[Double])])] =
   *         Array( params: Array(bins, depth, impurityNum, numTrees),
   *                modelMetrics: Array( (model, Array(fMeasure, precision, recall) ) ) )
   */
  def multiParamRfCvs( dataTrain:RDD[LabeledPoint], numClasses: Int, catInfo: Map[Int, Int],
    maxBins: Array[Int], maxDepths: Array[Int], impurities: Array[Int], maxTrees:Array[Int],
    numFolds: Int = 3)
  : Array[(Array[Int], Array[(RandomForestModel, Array[Double])])] = {
    val seed = 1
    val evaluations = for {
      impurityNum <- impurities
      depth <- maxDepths
      bins <- maxBins
      numTrees <- maxTrees
    } yield {
      val impurity = impurityNum match {
        case 0 => "gini"
        case _ => "entropy"
      }
      val folds: Array[(RDD[LabeledPoint], RDD[LabeledPoint])] = MLUtils.kFold(dataTrain, numFolds, seed)
      val modelMetrics = folds.
        map{ case (training, validation) =>
        training.cache;validation.cache()
        // training model
        println(f"bins=${bins}, depth=${depth}, impurity=${impurity}, trees=${numTrees}, ")
        val model = TimeEvaluation.time( RandomForest.trainClassifier(training, numClasses, catInfo, numTrees, "auto", impurity, depth, bins, seed) )
        training.unpersist()
        // validatiing model
        val predictLabels = model.predict( validation.map(_.features) ).
          zip( validation.map(_.label) )
        validation.unpersist()
        val metric = new MulticlassMetrics(predictLabels)
        ( model, Array(metric.fMeasure, metric.precision, metric.recall) ) }
      ( Array(bins, depth, impurityNum, numTrees), modelMetrics )
    }
    evaluations
  }


  def spam2Tf(sc:SparkContext, inpath:String, features:Int, lable:Double): RDD[LabeledPoint] = {
    val tf = new HashingTF(numFeatures = features)
    val dataRaw: RDD[(String, String)] = sc.wholeTextFiles(inpath)
    val dataLabeled: RDD[LabeledPoint] = tf.transform {
        dataRaw.map { case (file, text) => text.split(" ").toList }}.
      map{ features => LabeledPoint(lable, features) }
    dataLabeled
  }

  def prepareTf(sc:SparkContext, paths:Array[String], numFeatures: Int = 100, splits: Array[Double] = Array(0.8, 0.1, 0.1)): Array[RDD[LabeledPoint]] = {
    val Array(spampath, hampath) = paths
    val labeledSpam: RDD[LabeledPoint] = spam2Tf(sc, spampath, numFeatures, 1.0D)
    val labeledHam: RDD[LabeledPoint] = spam2Tf(sc, hampath, numFeatures, 0.0D)
    val dataset: Array[RDD[LabeledPoint]] = (labeledSpam.union(labeledHam)).randomSplit(splits)
    // val stats = Statistics.colStats{ dataCV.map{_.features} }
    dataset
  }

  def prepareTfIdf(rdds: Array[RDD[LabeledPoint]]): Array[RDD[LabeledPoint]] = {
    val Array(dataTrain, dataCV, dataTest) = rdds
    val idfModel: IDFModel = (new IDF()).fit{ (rdds.reduce((rdd1, rdd2) => rdd1.union(rdd2))).map{_.features} }
    val newRdds: Array[RDD[LabeledPoint]] = rdds.map{ rdd =>
      idfModel.transform( rdd.map(lp => lp.features) ).
        zip(rdd).
        map{ case (features, lp) => LabeledPoint(lp.label, features) }
    }
    newRdds
  }

  /**
   *
   * @param rdd: RDD[(score: Double, label: Double)]
   * @return (confusionMatrix: Array[Int], message: String)
   */
  def confusionMatrix(rdd: RDD[(Double, Double)]): (Array[Int], String) = {
    //
    val matrix = rdd.map{
      case (score, label) => (label, score) match {
        case (1.0, 1.0) => Array(1, 0, 0, 0) // TP
        case (0.0, 1.0) => Array(0, 1, 0, 0) // FP
        case (0.0, 0.0) => Array(0, 0, 1, 0) // TN
        case (1.0, 0.0) => Array(0, 0, 0, 1) // FN
      }
    }.reduce{
      (ary1, ary2) => Array(ary1(0)+ary2(0), ary1(1)+ary2(1), ary1(2)+ary2(2), ary1(3)+ary2(3))
    }
    //
    val eval = matrix.map(it => it.toDouble)
    val accuracy = 100*( ( eval(0) + eval(2) ) / (eval.sum) )
    val precision =100*(  eval(0)  / ( eval(0) + eval(1) ) )
    val recall =   100*(  eval(0) / ( eval(0) + eval(3) ) )
    val msg = f"accuracy = ${accuracy}%2.4f%%, precision = ${precision}%2.4f%%, recall = ${recall}%2.4f%%"
    //
    (matrix, msg)
  }

  /**
   * using models to predict results, and append the results to data
   * @param data: RDD[LabeledPoint]
   * @param models: Array[RandomForestModel]
   * @return RDD[LabeledPoint]
   */
  def appendPredictions(data: RDD[LabeledPoint], models: Array[RandomForestModel])
  : RDD[LabeledPoint] = {
    data.cache()
    val modelPredictions: Array[RDD[Double]] = models.map{ model => model.predict{ data.map{ _.features } } }
    var lps: RDD[(Double, Array[Double])] = data.map{ lp => (lp.label, lp.features.toArray) }
    modelPredictions.foreach{ predictions =>
      lps = lps.zip( predictions ).map{
        case ((label, features), prediction) => (label, features :+ prediction) } }
    val dataAndMeta = lps.map{ case (label, features) => LabeledPoint(label, Vectors.dense(features)) }
    data.unpersist()
    dataAndMeta
  }

}
