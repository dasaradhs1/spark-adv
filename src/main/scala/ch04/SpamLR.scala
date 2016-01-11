package ch04

import org.apache.spark.ml.Model
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.{Word2Vec, HashingTF}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql._
/**
 * Created by leo on 2015/12/11.
 */
object SpamLR {


  def main(args: Array[String]) {
    val appName = "Spam classification"
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)
    /*
    :load /media/sf_WORKSPACE.2W/spark-adv/src/main/scala/ch04/DTreeUtil.scala
    val args = Array("file:///media/sf_WORKSPACE.2W/dataset/spam/20030228.spam"
      , "file:///media/sf_WORKSPACE.2W/dataset/spam/20030228.easyham")

    :load /home/leoricklin/jar/DTreeUtil.scala
    val args = Array("file:///home/leoricklin/dataset/spam/20030228.spam"
      , "file:///home/leoricklin/dataset/spam/20030228.easyham")
     */
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    /**
     * MLlib TF featuring, LR model training, validating, parameters tuning
     * @return
     */
    val modelA1 = {
      // preparing data
      val Array(dataTrain, dataCV, dataTest) = DTreeUtil.prepareTf(sc, args, 100, Array(0.8, 0.1, 0.1))
      dataTrain.cache()
      dataCV.cache()
      dataTest.cache()
      /*
  dataTrain.first.features
  : Vector = (100,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,48,49,50,51,52,53,54,
       */
      // training model
      val lrModel = (new LogisticRegressionWithSGD()).run(dataTrain)
      /*
  : intercept = 0.0, numFeatures = 100, numClasses = 2, threshold = 0.5
       */
      lrModel.save(sc, "file:///home/leo/spam.model/")
      /*
  $ ls -l spam.model
  drwxrwxr-x 2 leo leo 4096  7æœˆ  8 17:48 data
  drwxrwxr-x 2 leo leo 4096  7æœˆ  8 17:48 metadata
       */
      // validating model
      val tf = new HashingTF(numFeatures = 100)
      val posTestExample: Vector = tf.transform{ "O M G GET cheap stuff by sending money to ...".split(" ") }
      val negTestExample: Vector = tf.transform{ "Hi Dad, I started studying Spark the other ...".split(" ") }
      println(s"Prediction for positive test example: ${lrModel.predict(posTestExample)}")
      println(s"Prediction for negative test example: ${lrModel.predict(negTestExample)}")
      /*
  Prediction for positive test example: 0.0
  Prediction for negative test example: 0.0
       */
      var scorewLabel = dataCV.
        zip{ lrModel.predict{ dataCV.map{_.features} } }.
        map{ case (lp, score) => (score, lp.label) }
      var metrics = new BinaryClassificationMetrics( scorewLabel )
      metrics.areaUnderROC() //  : Double = 0.6707152496626181
      DTreeUtil.confusionMatrix{ scorewLabel }
      /*
  = (Array(22, 11, 236, 35),accuracy = 84.8684%, precision = 66.6667%, recall = 38.5965%)
       */
      // tuning parameters
      lrModel.setThreshold(0.17)
      /*
  : intercept = 0.0, numFeatures = 100, numClasses = 2, threshold = 0.17
       */
      scorewLabel = dataCV.
        zip{ lrModel.predict{ dataCV.map{_.features} } }.
        map{ case (lp, score) => (score, lp.label) }
      metrics = new BinaryClassificationMetrics( scorewLabel )
      metrics.areaUnderROC() // : Double = 0.6707152496626181
      DTreeUtil.confusionMatrix{ scorewLabel }
      /*
       (Array(22, 11, 236, 35),accuracy = 84.8684%, precision = 66.6667%, recall = 38.5965%)
       */
    }

    /**
     * ML TF featuring, LR model training, validating
     */
    val modelA2 = {
      // preparing data
      val _label = "label"
      val _text = "text"
      val _words = "word"
      val _features = "features"
      val _prediction = "prediction"
      val _rawPrediction = "rawPrediction"
      val spam = sc.wholeTextFiles(args(0)).
        map{ case (file: String, text: String) => (1.0, text.toLowerCase()) }
      val ham = sc.wholeTextFiles(args(1)).
        map{ case (file: String, text: String) => (0.0, text.toLowerCase()) }
      val dataSet = (spam.union(ham)).
        toDF(_label,_text).
        randomSplit( Array(0.8,0.2) )
      dataSet.foreach(_.cache())
      // training model
      val tokenizer = new org.apache.spark.ml.feature.Tokenizer()
      val hashingTF = new org.apache.spark.ml.feature.HashingTF()
      val lr = new org.apache.spark.ml.classification.LogisticRegression()
      val paramMap = org.apache.spark.ml.param.ParamMap(
        tokenizer.inputCol ->   _text,
        tokenizer.outputCol ->  _words,
        hashingTF.inputCol ->   _words,
        hashingTF.outputCol ->  _features,
        hashingTF.numFeatures -> 100,
        lr.labelCol ->          _label,
        lr.featuresCol ->       _features,
        lr.predictionCol ->     _prediction,
        lr.rawPredictionCol ->  _rawPrediction,
        lr.probabilityCol ->    "myProbability",
        lr.maxIter -> 30,
        lr.regParam -> 0.1,
        lr.threshold -> 0.55)
      val model = lr.fit(
        hashingTF.transform(
          tokenizer.transform(dataSet(0), paramMap) ,
          paramMap),
        paramMap)
      /*
      model.extractParamMap : ParamMap =
{
        logreg_c1f860a4bc3d-elasticNetParam: 0.0,
        logreg_c1f860a4bc3d-featuresCol: features,
        logreg_c1f860a4bc3d-fitIntercept: true,
        logreg_c1f860a4bc3d-labelCol: label,
        logreg_c1f860a4bc3d-maxIter: 30,
        logreg_c1f860a4bc3d-predictionCol: prediction,
        logreg_c1f860a4bc3d-probabilityCol: myProbability,
        logreg_c1f860a4bc3d-rawPredictionCol: rawPrediction,
        logreg_c1f860a4bc3d-regParam: 0.1,
        logreg_c1f860a4bc3d-standardization: true,
        logreg_c1f860a4bc3d-threshold: 0.55,
        logreg_c1f860a4bc3d-tol: 1.0E-6
}
       */
      val evaluator = new org.apache.spark.ml.evaluation.BinaryClassificationEvaluator().
        setLabelCol(_label).
        setRawPredictionCol(_rawPrediction)
      evaluator.evaluate(
        model.transform(
          hashingTF.transform(
            tokenizer.transform(dataSet(1), paramMap) ,
          paramMap)))
      // : Double = 0.9993274799178031

    }
    /**
     * ML Cross validating with pipelines: tokenzing, TF featuring, LR modeling, n-folding
     */
    val modelA3 = {
      // preparing data
      val _label = "label"
      val _text = "text"
      val _words = "word"
      val _features = "features"
      val _prediction = "prediction"
      val _rawPrediction = "rawPrediction"
      val _probability = "probability"
      val spam = sc.wholeTextFiles(args(0)).
        map{ case (file: String, text: String) => (1.0, text.toLowerCase()) }
      val ham = sc.wholeTextFiles(args(1)).
        map{ case (file: String, text: String) => (0.0, text.toLowerCase()) }
      val dataSet = (spam.union(ham)).
        toDF(_label,_text).
        randomSplit( Array(0.8,0.2) )
      dataSet.foreach(_.cache())
      // training model
      val tokenizer = new org.apache.spark.ml.feature.Tokenizer().
        setInputCol(_text).
        setOutputCol(_words)
      val hashingTF = new org.apache.spark.ml.feature.HashingTF().
        setInputCol(_words).
        setOutputCol(_features)
      val lr = new org.apache.spark.ml.classification.LogisticRegression().
        setLabelCol(_label).
        setFeaturesCol(_features).
        setPredictionCol(_prediction).
        setProbabilityCol(_probability).
        setRawPredictionCol(_rawPrediction)
      val evaluator = new org.apache.spark.ml.evaluation.BinaryClassificationEvaluator().
        setLabelCol(_label).
        setRawPredictionCol(_rawPrediction)
      /*
       evaluator.extractParamMap : ParamMap =
{
        binEval_05e104ad54cf-labelCol: label,
        binEval_05e104ad54cf-metricName: areaUnderROC,
        binEval_05e104ad54cf-rawPredictionCol: rawPrediction
}
       */
      val paramGrid = new org.apache.spark.ml.tuning.ParamGridBuilder().
        addGrid( hashingTF.numFeatures, Array(100, 1000) ).
        addGrid( lr.maxIter,          Array(10, 20) ).
        addGrid( lr.regParam,         Array(0.1, 0.01) ).
        addGrid( lr.threshold,        Array(0.55, 0.65) ).
        build()
      val pipeline = new org.apache.spark.ml.Pipeline().
        setStages( Array(tokenizer, hashingTF, lr) )
      val crossval = new org.apache.spark.ml.tuning.CrossValidator().
        setEstimator(pipeline).
        setEstimatorParamMaps(paramGrid).
        setEvaluator(evaluator).
        setNumFolds(3)
      val model = crossval.fit(dataSet(0))
      /*
      model.bestModel.extractParamMap() : ParamMap =
      {
      }
      model.extractParamMap().get(model.estimatorParamMaps).get.foreach(println)
{
        logreg_3d36717c185d-maxIter: 10,
        hashingTF_21f034d6bef3-numFeatures: 100,
        logreg_3d36717c185d-regParam: 0.1,
        logreg_3d36717c185d-threshold: 0.55
}
{
        logreg_3d36717c185d-maxIter: 20,
        hashingTF_21f034d6bef3-numFeatures: 100,
        logreg_3d36717c185d-regParam: 0.1,
        logreg_3d36717c185d-threshold: 0.55
}
{
        logreg_3d36717c185d-maxIter: 10,
        hashingTF_21f034d6bef3-numFeatures: 100,
        logreg_3d36717c185d-regParam: 0.1,
        logreg_3d36717c185d-threshold: 0.65
}
{
        logreg_3d36717c185d-maxIter: 20,
        hashingTF_21f034d6bef3-numFeatures: 100,
        logreg_3d36717c185d-regParam: 0.1,
        logreg_3d36717c185d-threshold: 0.65
}
{
        logreg_3d36717c185d-maxIter: 10,
        hashingTF_21f034d6bef3-numFeatures: 100,
        logreg_3d36717c185d-regParam: 0.01,
        logreg_3d36717c185d-threshold: 0.55
}
{
        logreg_3d36717c185d-maxIter: 20,
        hashingTF_21f034d6bef3-numFeatures: 100,
        logreg_3d36717c185d-regParam: 0.01,
        logreg_3d36717c185d-threshold: 0.55
}
{
        logreg_3d36717c185d-maxIter: 10,
        hashingTF_21f034d6bef3-numFeatures: 100,
        logreg_3d36717c185d-regParam: 0.01,
        logreg_3d36717c185d-threshold: 0.65
}
{
        logreg_3d36717c185d-maxIter: 20,
        hashingTF_21f034d6bef3-numFeatures: 100,
        logreg_3d36717c185d-regParam: 0.01,
        logreg_3d36717c185d-threshold: 0.65
}
{
        logreg_3d36717c185d-maxIter: 10,
        hashingTF_21f034d6bef3-numFeatures: 1000,
        logreg_3d36717c185d-regParam: 0.1,
        logreg_3d36717c185d-threshold: 0.55
}
{
        logreg_3d36717c185d-maxIter: 20,
        hashingTF_21f034d6bef3-numFeatures: 1000,
        logreg_3d36717c185d-regParam: 0.1,
        logreg_3d36717c185d-threshold: 0.55
}
{
        logreg_3d36717c185d-maxIter: 10,
        hashingTF_21f034d6bef3-numFeatures: 1000,
        logreg_3d36717c185d-regParam: 0.1,
        logreg_3d36717c185d-threshold: 0.65
}
{
        logreg_3d36717c185d-maxIter: 20,
        hashingTF_21f034d6bef3-numFeatures: 1000,
        logreg_3d36717c185d-regParam: 0.1,
        logreg_3d36717c185d-threshold: 0.65
}

       */
      // testing model
      val res = model.transform(dataSet(1))
      /*
      : DataFrame = [label: double, text: string, words: array<string>, features: vector
      , rawPrediction: vector, probability: vector, prediction: double]
       */
      res.select(_label,_prediction,_probability,_rawPrediction).take(5)
      /*
      : Array[Row] = Array(
      [1.0,1.0,[0.15103944287192522,0.8489605571280747],[-1.7264717134192873,1.7264717134192873]],
      [1.0,1.0,[0.0015236531571747208,0.9984763468428254],[-6.485119619678206,6.485119619678206]],
      [1.0,1.0,[0.17938343098000914,0.8206165690199908],[-1.5205303837635542,1.5205303837635542]],
      [1.0,1.0,[0.05971644480015286,0.9402835551998471],[-2.7565740444872553,2.7565740444872553]],
      [1.0,1.0,[2.801181453089618E-4,0.9997198818546911],[-8.18001893887212,8.18001893887212]])
       */
      evaluator.evaluate(res)
      // : Double = 0.999978250467615
      DTreeUtil.confusionMatrix(
        res.select(_label,_prediction).
          map{ case Row(label: Double, score: Double) => (score, label) })
      /*
      (Array[Int], String) = (Array(88, 0, 474, 9),accuracy = 98.4238%, precision = 100.0000%, recall = 90.7216%)
       */
    }

    /**
     * Removing HTML tag, TF featuring, LR model training, validating, parameters tuning
     * @return
     */
    val modelA4 = {
      case class Email(text:String)
      case class EmailLabeled(text:String, label:Double)
      val spamTrain = sc.wholeTextFiles("file:///home/leo/spam/20030228.spam").map {
        case (file, content) =>
          Email( HtmlUtil.removePunct(
            HtmlUtil.removeConcateTag(
              HtmlUtil.removeHtmlTag(
                HtmlUtil.getLoCaseHtmlBody(HtmlUtil.removeLF(content, "")), " "), ""), "") )
      }.map{ case Email(text) => EmailLabeled(text, 1.0) }
      val hamTrain = sc.wholeTextFiles("file:///home/leo/spam/20030228.easyham").map {
        case (file, content) =>
          Email( HtmlUtil.removePunct(
            HtmlUtil.removeConcateTag(
              HtmlUtil.removeHtmlTag(
                HtmlUtil.getLoCaseHtmlBody(HtmlUtil.removeLF(content, "")), " "), ""), "") )
      }.map{ case Email(text) => EmailLabeled(text, 0.0) }
      val sampleSet = (spamTrain ++ hamTrain).toDF()
      sampleSet.cache()
      val trainSet = sampleSet.sample(false, 0.85, 100L)
      val testSet = sampleSet.sample(false, 0.15, 100L)
      /*
      sampleSet.count
      sampleSet.printSchema
      root
      |-- text: string (nullable = true)
      |-- label: double (nullable = false)

      trainSet.count = 2528
      trainSet.filter("label = 1.0").count() = 421
      trainSet.filter($"label" === 0.0).count() = 2107

      testSet.count = 437
      testSet.filter("label = 1.0").count() = 84
      testSet.filter($"label" === 0.0).count() = 353
      353/437D = 80.7780%
       */
      val tokenizer = new org.apache.spark.ml.feature.Tokenizer().
        setInputCol("text").
        setOutputCol("words")
      val hashingTF = new org.apache.spark.ml.feature.HashingTF().
        setInputCol(tokenizer.getOutputCol).
        setOutputCol("features")
      val lr = new org.apache.spark.ml.classification.LogisticRegression().
        setMaxIter(10)
      val pipeline = new org.apache.spark.ml.Pipeline().
        setStages(Array(tokenizer, hashingTF, lr))
      val crossval = new org.apache.spark.ml.tuning.CrossValidator().
        setEstimator(pipeline).
        setEvaluator(new org.apache.spark.ml.evaluation.BinaryClassificationEvaluator)
      val paramGrid = new org.apache.spark.ml.tuning.ParamGridBuilder().
        addGrid( hashingTF.numFeatures, Array(10, 100, 1000)).
        addGrid( lr.regParam, Array(0.1, 0.01)).
        addGrid( lr.maxIter, Array(10, 20, 30, 50)).
        build()
      crossval.setEstimatorParamMaps(paramGrid).setNumFolds(3)
      val cvModel = crossval.fit(trainSet)
      /*
      cvModel.bestModel.fittingParamMap: org.apache.spark.ml.param.ParamMap =
      {
          LogisticRegression-3cb51fc7-maxIter: 20,
          HashingTF-cb518e45-numFeatures: 1000,
          LogisticRegression-3cb51fc7-regParam: 0.1,
          Pipeline-ce5dacdb-stages: [Lorg.apache.spark.ml.PipelineStage;@f44062
      }
       */
      val validation = cvModel.transform(testSet)
      /*
      validation: org.apache.spark.sql.DataFrame = [
         text: string, label: double, words: array<string>, features: vector, rawPrediction: vector
       , probability: vector, prediction: double]
       */
      val matrix = validation.select("label","prediction").map{
        case Row(label: Double, prediction: Double) => (label, prediction) match {
          case (1.0, 1.0) => Array(1, 0, 0, 0) // TP
          case (0.0, 1.0) => Array(0, 1, 0, 0) // FP
          case (0.0, 0.0) => Array(0, 0, 1, 0) // TN
          case (1.0, 0.0) => Array(0, 0, 0, 1) // FN
        }
      }.reduce{
        (ary1, ary2) => Array(ary1(0)+ary2(0), ary1(1)+ary2(1), ary1(2)+ary2(2), ary1(3)+ary2(3))
      }
      /*
      Array(83, 0, 353, 1)
      evaluation(matrix)
       accuracy = 99.7712%, precision = 100.0000%, recall = 98.8095%
       */
    }

    /**
     * TFIDF featuring, LR model training, validating, parameters tuning
     */
    val modelB1 = {
      // preparing data
      val Array(dataTrain, dataCV, dataTest) = DTreeUtil.prepareTfIdf(DTreeUtil.prepareTf(sc, args, 100, Array(0.8, 0.1, 0.1)))
      dataTrain.cache()
      dataCV.cache()
      dataTest.cache()
      val lrModel = (new LogisticRegressionWithSGD()).run(dataTrain)
      // LogisticRegressionModel: intercept = 0.0, numFeatures = 100, numClasses = 2, threshold = 0.5
      val scorewLabel: RDD[(Double, Double)] = lrModel.predict{ dataCV.map{_.features} }.
        zip(dataCV).
        map{ case (score, lp) => (score, lp.label) }
      var metrics = new BinaryClassificationMetrics( scorewLabel )
      metrics.areaUnderROC() //  : Double = 0.731854480922804
      metrics = new BinaryClassificationMetrics( scorewLabel, 100 )
      metrics.areaUnderROC() //  : Double = 0.731854480922804
      DTreeUtil.confusionMatrix{ scorewLabel }
      // (Array(24, 6, 224, 25),accuracy = 88.8889%, precision = 80.0000%, recall = 48.9796%)
    }

    /**
     * Word2Vector
     */
    val modelC1 = {
      // preparing data
      val spam: RDD[(Double, List[String])] = sc.wholeTextFiles(args(0)).
        map{ case (file: String, text: String) =>
        (1.0, text.toLowerCase().split("""\s+""").toList)}
      val ham: RDD[(Double, List[String])] = sc.wholeTextFiles(args(1)).
        map{ case (file: String, text: String) =>
        (0.0, text.toLowerCase().split("""\s+""").toList)}
      val Array(trainset, cvset, testset) = (spam.union(ham)).randomSplit(Array(0.8,0.1,0.1))
      trainset.cache()
      trainset.first
      /*
(Double, List[String]) = (1.0,List(from, ilug-admin@linux.ie, mon, aug, 26, 15:48:46, 2002, return-path:, <ilug-admin@linux.ie>, delivered-to:, zzzz@localhost.spamassassin.taint.org, received:, from, localhost, (localhost, [127.0.0.1]), by, phobos.labs.spamassassin.taint.org, (postfix), with, esmtp, id, 9e40d44166, for, <zzzz@localhost>;, mon,, 26, aug, 2002, 10:41:42, -0400, (edt), received:, from, phobos, [127.0.0.1], by, localhost, with, imap, (fetchmail-5.9.0), for, zzzz@localhost, (single-drop);, mon,, 26, aug, 2002, 15:41:42, +0100, (ist), received:, from, lugh.tuatha.org, (root@lugh.tuatha.org, [194.125.145.45]), by, dogma.slashnull.org, (8.11.6/8.11.6), with, esmtp, id, g7nm2tz09890, for, <zzzz-ilug@jmason.org>;, fri,, 23, aug, 2002, 23:02:29, +0100, received:, from, lugh.
       */
      cvset.cache()
      testset.cache()

      // training model
      val w2vModel = (new Word2Vec()).fit{
        Array(trainset, cvset, testset).
          map{ rdd => rdd.map{ case(label, words) => words } }.
          reduce((rdd1, rdd2) => rdd1.union(rdd2))
      }
      w2vModel.findSynonyms("ilug-admin@linux.ie", 5)
      /*
      Array[(String, Double)] = Array((<rick@linuxmafia.com>,2.369212427535932), (iiu-admin@taint.org,2.2231481930351276), (irregulars-admin@tb.tf,2.1752848603260846), (social-admin@linux.ie,2.174064282289761), (fork-admin@xent.com,2.0671033197118573))
       */
      w2vModel.findSynonyms("<zzzz@localhost>", 5)
      /*
      java.lang.IllegalStateException: <zzzz@localhost> not in vocabulary
       */
      // validating model
    }
  }

}
