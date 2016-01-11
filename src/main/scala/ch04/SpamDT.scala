package ch04

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by leo on 2015/8/24.
 */
object SpamDT {
  val appName = "Basic Decision Tree for Spam classification"
  val sparkConf = new SparkConf().setAppName(appName)
  val sc = new SparkContext(sparkConf)
  // :load /media/sf_WORKSPACE.2L/spark-adv/src/main/scala/ch04/DTreeUtil.scala
  // :load /media/sf_WORKSPACE.2L/spark-util/src/main/scala/tw/com/chttl/spark/test/util/TimeEvaluation.scala
  // val args = Array("file:///media/sf_WORKSPACE.2L/dataset/spam/20030228.spam", "file:///media/sf_WORKSPACE.2L/dataset/spam/20030228.easyham")

  def dTSimple(sc:SparkContext,  args: Array[String]) = {
    val Array(spampath, hampath) = args
    val labeledSpam = DTreeUtil.spam2Tf(sc, spampath, 100, 1.0D)
    val labeledHam = DTreeUtil.spam2Tf(sc, hampath, 100, 0.0D)
    labeledSpam.cache();labeledHam.cache()
    val Array(dataTrain, dataCV, dataTest) = (labeledSpam.union(labeledHam)).randomSplit(Array(0.8, 0.1, 0.1))
    /*
    dataTrain.count = 2378
    dataCV.count = 320
    dataTest.count = 293
     */
    val model: DecisionTreeModel = DecisionTree.trainClassifier(dataTrain, 2, Map[Int,Int](), "gini", 4, 100)
    val predictionsAndLabels = dataCV.map(example =>
      (model.predict(example.features), example.label) )
    val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
    val ret = metrics.roc().map{ case (fpr, tpr) =>
      f"${fpr}}\t${tpr}"
    }.collect().mkString("\n")
    println("%table fpr\ttpr\n" + ret)
    /*
    val metrics: MulticlassMetrics = DTreeUtil.getMetrics(model, dataCV)
    metrics.precision = 0.8821548821548821
    metrics.recall = 0.8821548821548821
    metrics.confusionMatrix: Matrix =
     A\P |   0     1
     ----+-------------
      0  | 234.0  10.0
      1  | 25.0   28.0
    val Array(tn,fn,fp,tp) = metrics.confusionMatrix.toArray : Array[Double]
     = Array(234.0, 25.0, 10.0, 28.0)

    val accuracy = (tp+tn)/(tn+fn+fp+tp).toDouble = 0.8821548821548821
    val precision = tp/(tp+fp).toDouble = 0.7368421052631579
    val recall = tp/(tp+fn).toDouble = 0.5283018867924528

    val metric:Array[Int] = dataCV.map{ lp => (lp.label, model.predict(lp.features)) }.map{ ret =>
      ret match {
        case (1.0, 1.0) => Array(1, 0, 0, 0) // TP
        case (0.0, 1.0) => Array(0, 1, 0, 0) // FP
        case (0.0, 0.0) => Array(0, 0, 1, 0) // TN
        case (1.0, 0.0) => Array(0, 0, 0, 1) // FN
      }
    }.reduce{
      (ary1, ary2) => Array(ary1(0)+ary2(0), ary1(1)+ary2(1), ary1(2)+ary2(2), ary1(3)+ary2(3))
    }
     = Array(28, 10, 234, 25)
     */
    Array((model, ("gini", 4, 100), metrics))
  }

  def dTMultiHyperParam(sc:SparkContext,  args: Array[String]) = {
    val Array(spampath, hampath) = args
    val labeledSpam = DTreeUtil.spam2Tf(sc, spampath, 1000, 1.0D)
    val labeledHam = DTreeUtil.spam2Tf(sc, hampath, 1000, 0.0D)
    labeledSpam.cache();labeledHam.cache()
    val Array(dataTrain, dataCV, dataTest) = (labeledSpam.union(labeledHam)).randomSplit(Array(0.8, 0.1, 0.1))
    val evaluations = (for {
      impurity <- Array("gini", "entropy")
      depth <- Array(10, 20, 30)
      bins <- Array(40, 300)
    } yield {
        val model = DecisionTree.trainClassifier(dataTrain
          , 2, Map[Int,Int](), impurity, depth, bins)
        val predictionsAndLabels = dataCV.map(example =>
          (model.predict(example.features), example.label) )
        val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
        (model, (impurity, depth, bins), metrics)
    }).sortBy{ case (model, (impurity, depth, bins), metrics) =>
        metrics.areaUnderROC }.reverse

    evaluations.map{ case (model, (impurity, depth, bins), metrics) =>
      val cnfmatrix = metrics.scoreAndLabels.map{ sl => sl match {
        case (1.0, 1.0) => Array(1, 0, 0, 0) // TP
        case (0.0, 1.0) => Array(0, 1, 0, 0) // FN
        case (0.0, 0.0) => Array(0, 0, 1, 0) // TN
        case (1.0, 0.0) => Array(0, 0, 0, 1) // FP
      }}.reduce{ (ary1, ary2) => Array(
        ary1(0)+ary2(0), ary1(1)+ary2(1), ary1(2)+ary2(2), ary1(3)+ary2(3))
      }
      val accuracy = (cnfmatrix(0)+cnfmatrix(2)) / (cnfmatrix.sum).toDouble
      s"impurity=${impurity}, depth=${depth}, bins=${bins}), AUC=${metrics.areaUnderROC}, accuracy=${accuracy}"
    }.foreach(println)
    /*
impurity=entropy, depth=30, bins=40),  AUC=0.8523409363745498, accuracy=0.9163763066202091
impurity=entropy, depth=20, bins=40),  AUC=0.8523409363745498, accuracy=0.9163763066202091
impurity=entropy, depth=10, bins=40),  AUC=0.8523409363745498, accuracy=0.9163763066202091
impurity=entropy, depth=10, bins=300), AUC=0.8319327731092437, accuracy=0.9094076655052264
impurity=gini,    depth=20, bins=300), AUC=0.8319327731092437, accuracy=0.9094076655052264
impurity=gini,    depth=30, bins=300), AUC=0.8298319327731092, accuracy=0.9059233449477352
impurity=entropy, depth=30, bins=300), AUC=0.8277310924369748, accuracy=0.9024390243902439
impurity=entropy, depth=20, bins=300), AUC=0.8277310924369748, accuracy=0.9024390243902439
impurity=gini,    depth=30, bins=40),  AUC=0.8172268907563027, accuracy=0.8850174216027874
impurity=gini,    depth=20, bins=40),  AUC=0.8112244897959183, accuracy=0.8885017421602788
impurity=gini,    depth=10, bins=300), AUC=0.8013205282112843, accuracy=0.8989547038327527
impurity=gini,    depth=10, bins=40),  AUC=0.7827130852340937, accuracy=0.8815331010452961
     */


    /*
    val evaluations: Array[(DecisionTreeModel, (String, Int, Int), MulticlassMetrics)] = DTreeUtil.multiParamModels(
      dataTest, dataCV, 2, Map[Int,Int](), Array("gini", "entropy"), Array(10, 20, 30), Array(40, 300))
    evaluations
    val output = evaluations.map{ case (model, (impurity, depth, bins), metrics) =>
      val Array(tn,fn,fp,tp) = metrics.confusionMatrix.toArray
      val accuracy = (tp+tn)/(tn+fn+fp+tp)
      val precision = (tp)/(tp+fp)
      val recall = (tp)/(tp+fn)
      s"impurity=${impurity}, depth=${depth}, bins=${bins}), accuracy=${accuracy}, precision=${precision}, recall=${recall}"
    }
     output.foreach(println)
impurity=gini, depth=30, bins=300), accuracy=0.9311475409836065, precision=0.9444444444444444, recall=0.6415094339622641
impurity=gini, depth=30, bins=40),  accuracy=0.9311475409836065, precision=0.9444444444444444, recall=0.6415094339622641
impurity=gini, depth=20, bins=300), accuracy=0.9311475409836065, precision=0.9444444444444444, recall=0.6415094339622641
impurity=gini, depth=20, bins=40),  accuracy=0.9311475409836065, precision=0.9444444444444444, recall=0.6415094339622641
impurity=gini, depth=10, bins=300), accuracy=0.9311475409836065, precision=0.9444444444444444, recall=0.6415094339622641
impurity=gini, depth=10, bins=40),  accuracy=0.9311475409836065, precision=0.9444444444444444, recall=0.6415094339622641
impurity=entropy, depth=30, bins=300), accuracy=0.921311475409836, precision=0.8085106382978723, recall=0.7169811320754716
impurity=entropy, depth=30, bins=40), accuracy=0.921311475409836, precision=0.8085106382978723, recall=0.7169811320754716
impurity=entropy, depth=20, bins=300), accuracy=0.921311475409836, precision=0.8085106382978723, recall=0.7169811320754716
impurity=entropy, depth=20, bins=40), accuracy=0.921311475409836, precision=0.8085106382978723, recall=0.7169811320754716
impurity=entropy, depth=10, bins=300), accuracy=0.921311475409836, precision=0.8085106382978723, recall=0.7169811320754716
impurity=entropy, depth=10, bins=40), accuracy=0.921311475409836, precision=0.8085106382978723, recall=0.7169811320754716
     */
  }

  def main(args: Array[String]) {
    if (args.length != 1) {
      System.err.printf("Arguments:%s\n", args.mkString("[",",","]"))
      System.err.printf("Usage: %s <input_path>", appName)
      System.exit(1)
    }
    var evalutions = dTSimple(sc, args)

  }
}
