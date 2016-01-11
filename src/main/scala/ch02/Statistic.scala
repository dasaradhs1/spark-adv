package ch02

import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.{SparkContext, SparkConf}
import tw.com.chttl.spark.mllib.util.{NAStatCounter, NAStat}
import scala.collection.immutable.IndexedSeq


/**
 * Created by leorick on 2015/12/21.
 */
object Statistic {
  def main(args: Array[String]) {
    val appName = "Spam classification"
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)

    // Correlation
    val model1 = {
      /*
      reference: http://www.real-statistics.com/correlation/multiple-correlation/
Poverty	Infant Mort	White	Crime	Doctors	Traf Deaths	University	Unemployed	Income
Alabama	15.7	9.0	71.0	448	218.2	1.81	22.0	5.0	42,666
Alaska	8.4	6.9	70.6	661	228.5	1.63	27.3	6.7	68,460
Arizona	14.7	6.4	86.5	483	209.7	1.69	25.1	5.5	50,958
...
       */
      val src = sc.textFile("file:///home/leoricklin/dataset/correlation.txt").map(line => line.split("""\s+"""))
      val vecs = src.map{ _.map(_.replaceAll("""\D+""","")) }.
        map{_.map(_.toDouble)}.
        map{ Vectors.dense(_) }
      val res: Matrix = Statistics.corr(vecs, "pearson")
      val rows = res.numRows
      val cols = res.numCols
      val ary = res.toArray
      val coef: IndexedSeq[Array[Double]] = (0 to rows).map{ idx => ary.slice(idx*cols, (idx*cols)+cols) }
      val titles: IndexedSeq[String] = (1 to 9).map( id => id.toString() + " "*5 )
      val out = f"${" "*6}|${titles.mkString("|")}\n" ++ {
      for (i <- 0 until titles.size) yield {
        f"${titles(i)}|${coef(i).map(v => f"${v}%+1.3f").mkString("|")}"
      }}.mkString("\n")
/*
println(out)
      |1     |2     |3     |4     |5     |6     |7     |8     |9
1     |+1.000|+0.564|-0.112|+0.275|-0.428|+0.673|-0.727|+0.281|-0.835
2     |+0.564|+1.000|-0.381|+0.428|-0.327|+0.562|-0.587|+0.228|-0.489
3     |-0.112|-0.381|+1.000|-0.427|-0.124|-0.163|-0.003|-0.171|-0.232
4     |+0.275|+0.428|-0.427|+1.000|-0.094|+0.314|-0.220|+0.382|-0.050
5     |-0.428|-0.327|-0.124|-0.094|+1.000|-0.641|+0.720|+0.115|+0.588
6     |+0.673|+0.562|-0.163|+0.314|-0.641|+1.000|-0.763|-0.058|-0.675
7     |-0.727|-0.587|-0.003|-0.220|+0.720|-0.763|+1.000|-0.091|+0.821
8     |+0.281|+0.228|-0.171|+0.382|+0.115|-0.058|-0.091|+1.000|-0.016
9     |-0.835|-0.489|-0.232|-0.050|+0.588|-0.675|+0.821|-0.016|+1.000
 */
    }

    // Pearson's independence test, 參考 R 軟體與績統計分析班 (CB1F0P001), p.091, 獨立性檢定
    val model2 = {
      val data = Matrices.dense(2,4, Array(

           12813, 65963  , 647   , 4000
        ,  359,   2642,    42,     303))
      /*
      : Matrix =
      // None,  Minimal, Minor, Major
       12813.0  647.0   359.0   42.0    // Belt Yes
       65963.0  4000.0  2642.0  303.0   // Belt No
       */

      var stats1 = Statistics.chiSqTest(data)
      /*
      : ChiSqTestResult =
       Chi squared test summary:
       method: pearson
       degrees of freedom = 3
       statistic = 59.22397368209237
       pValue = 8.610889778992714E-13
       Very strong presumption against null hypothesis: the occurrence of the outcomes is statistically independent..
       */

      val data2 = sc.parallelize{ List(
        LabeledPoint( 0, Vectors.dense(Array(12813.0, 65963.0)) ),
        LabeledPoint( 1, Vectors.dense(Array(647.0,   4000.0)) ),
        LabeledPoint( 2, Vectors.dense(Array(359.0,   2642.0)) ),
        LabeledPoint( 3, Vectors.dense(Array(42.0,    303.0)) )
        )}
      var stats2 = Statistics.chiSqTest(data2)
      stats2.foreach(stat => println(f"----------\n${stat}"))
      /*
----------
Chi squared test summary:
method: pearson
degrees of freedom = 9
statistic = 12.0
pValue = 0.21330930508341628
No presumption against null hypothesis: the occurrence of the outcomes is statistically independent..
----------
Chi squared test summary:
method: pearson
degrees of freedom = 9
statistic = 12.0
pValue = 0.21330930508341628
No presumption against null hypothesis: the occurrence of the outcomes is statistically independent..

       */
    }

    // Pearson's chi-squared goodness of fit test, 參考 R 軟體與績統計分析班 (CB1F0P001), p.090, 適合度檢定
    val model3 = {
      val freq = Vectors.dense(100,110,80,55,14)  // the observed data
      val prob = Vectors.dense(29,21,17,17,16)    // the expected distribution
      val stats = Statistics.chiSqTest(freq, prob)
      /*
      : ChiSqTestResult =
       Chi squared test summary:
       method: pearson
       degrees of freedom = 4
       statistic = 55.39549501483428
       pValue = 2.6845303757738748E-11
       Very strong presumption against null hypothesis: observed follows the same distribution as expected..
       */
      val freq2 = Vectors.dense(19,28,34,21,25)
      val stats2 = Statistics.chiSqTest(freq2)
      /* 參考 R 軟體與績統計分析班 (CB1F0P001), p.090, 適合度檢定
      : ChiSqTestResult =
       Chi squared test summary:
       method: pearson
       degrees of freedom = 4
       statistic = 5.559055118110236
       pValue = 0.23458705070741892
       No presumption against null hypothesis: observed follows the same distribution as expected..
       */

    }

  }


}
