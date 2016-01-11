package ch03

import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.feature.{StandardScalerModel, StandardScaler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.recommendation.{ALSModel, ALS}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.{SparkContext, SparkConf}
import tw.com.chttl.spark.core.util._
/**
 * Created by leorick on 2015/11/30.
 */
object RecALSPipeLine {
  val appName = "ALS Recommendation with ML Pipeline"
  val sparkConf = new SparkConf().setAppName(appName)
  val sc = new SparkContext(sparkConf)
  //
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._
  case class ArtistAlias(artAliasId:Long, artistId:Long)
  case class UserPlayList(userId:Long, usrAliasId:Long, count:Long)
  case class ArtistData(datArtistId:Long, artistName:String)

  /**
   * compute the root meas square error
   * @param predictions: RDD[(rating: Double, prediction: Double)]
   * @return the rmse
   * e.g.: computeRmse{ sc.parallelize(Array((2.0, 3.0),(3.0, 5.0),(4.0, 7.0))) }  = 2.160246899469287
   */
  def computeRmse(predictions: RDD[(Double, Double)]) = {
    val dataCnt = predictions.count
    val rmse = math.
      sqrt(predictions.map{
        case (rating: Double, prediction: Double) => (rating - prediction)*(rating - prediction) }.
      reduce(_+_) / dataCnt )
    rmse
  }

  def parseArtAli(): DataFrame = {
    val rawArtAli = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/artist_alias.txt")
    rawArtAli.flatMap{ line =>
      val tokens = line.split('\t') // tokenize
      if (tokens.size != 2 || tokens(0).isEmpty || tokens(1).isEmpty) {
        None
      } else {
        Some( ArtistAlias(tokens(0).toLong, tokens(1).toLong) )
      }}.
    toDF() // : DataFrame = [artAliasId: bigint, artistId: bigint]
  }

  def parseUserPlay(): DataFrame = {
    val rawUA = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/user_artist_data.txt")
    rawUA.map{ line =>
      val tokens = line.split(' ')
      UserPlayList(tokens(0).toLong, tokens(1).toLong, tokens(2).toLong)
    }.toDF() // : DataFrame =  [userId: bigint, usrAliasId: bigint, count: bigint]
  }

  def parseArtName(): DataFrame = {
    val rawArtData = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/artist_data.txt")
    rawArtData.flatMap { line =>
      val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      } else {
        try {
          Some(ArtistData(id.toLong, name.trim))
        } catch {
          case e: NumberFormatException => None
        }
      }
    }.toDF() // : DataFrame = [datArtistId: bigint, artistName: string]
  }

  def model1 = {
    /* load artist alias
    NAStat.statsWithMissing( rawArtAli.map{line => Array(line.split('\t').size.toDouble)} ) : Array[NAStatCounter] = Array(
     stats: (count: 193027, mean: 2.000000, stdev: 0.000000, max: 2.000000, min: 2.000000), NaN: 0)
     */
    val artistAlias = parseArtAli()
    /* load user playlist
    NAStat.statsWithMissing( rawUA.map{ line => Array(line.split(' ').size.toDouble)} ) : Array[NAStatCounter] = Array(
     stats: (count: 24296858, mean: 3.000000, stdev: 0.000000, max: 3.000000, min: 3.000000), NaN: 0)
     */
    val userPlayList = parseUserPlay()
    /* load artist name
    NAStat.statsWithMissing( rawArtData.map{ line => Array(line.split('\t').size.toDouble)} ) : Array[NAStatCounter] = Array(
    stats: (count: 1848707, mean: 1.999860, stdev: 0.019611, max: 9.000000, min: 1.000000), NaN: 0)
     */
    val artistData = parseArtName()
    // preparing data
    val userRating = userPlayList.
      join(artistAlias, userPlayList("usrAliasId") === artistAlias("artAliasId"), "left_outer").
      selectExpr("userId","nvl(artistId, usrAliasId)","count").
      map{ case Row(userId: Long, artistId: Long, count: Long) =>
        (userId.toInt, artistId.toInt, count.toDouble)}.
      toDF("uid", "pid", "rating") // : DataFrame = [uid: int, pid: int, rating: double]
    val Array(trainData, cvData, testData) = userRating.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache() // cvData.count() = 2429720
    testData.cache()
    /*
    cvData.describe("rating").show()
+-------+------------------+
|summary|            rating|
+-------+------------------+
|  count|           2430182|
|   mean|15.229228921949055|
| stddev|  96.8352348860034|
|    min|               1.0|
|    max|          101076.0|
+-------+------------------+
    testData.describe("rating").show()
+-------+------------------+
|summary|            rating|
+-------+------------------+
|  count|           2428420|
|   mean|15.472853130842276|
| stddev| 293.7031816022673|
|    min|               1.0|
|    max|          439771.0|
+-------+------------------+
     */
    // standardize parameters
    val scaler = new StandardScaler().setInputCol("rating").setOutputCol("std_rating")
    val scalerModel = scaler.
      fit{ trainData.
        map{ case Row(uid: Int, pid: Int, rating: Double) => (uid, pid, Vectors.dense(rating.toDouble))}.
        toDF("uid", "pid", "rating") }
    val trainDataStd = scalerModel.
      transform{ trainData.
        map{ case Row(uid: Int, pid: Int, rating: Double) => (uid, pid, Vectors.dense(rating.toDouble)) }.
        toDF("uid", "pid", "rating") }.
      map{ case Row(uid: Int, pid: Int, rating: Vector, std_rating: Vector) => (uid, pid, rating.toArray.head, std_rating.toArray.head) }.
      toDF("uid", "pid", "rating", "std_rating")
    val testDataStd = scalerModel.
      transform{ testData.
        map{ case Row(uid: Int, pid: Int, rating: Double) => (uid, pid, Vectors.dense(rating.toDouble)) }.
        toDF("uid", "pid", "rating") }.
      map{ case Row(uid: Int, pid: Int, rating: Vector, std_rating: Vector) => (uid, pid, rating.toArray.head, std_rating.toArray.head) }.
      toDF("uid", "pid", "rating", "std_rating")
    val cvDataStd = scalerModel.
      transform{ cvData.
        map{ case Row(uid: Int, pid: Int, rating: Double) => (uid, pid, Vectors.dense(rating.toDouble)) }.
        toDF("uid", "pid", "rating") }.
      map{ case Row(uid: Int, pid: Int, rating: Vector, std_rating: Vector) => (uid, pid, rating.toArray.head, std_rating.toArray.head) }.
      toDF("uid", "pid", "rating", "std_rating")
    /*
    trainDataStd.describe("rating", "std_rating").show()
+-------+------------------+--------------------+
|summary|            rating|          std_rating|
+-------+------------------+--------------------+
|  count|          19438256|            19438256|
|   mean|15.281956622034405| 0.11498889144629124|
| stddev| 132.8994134771187|  0.9999999742281268|
|    min|               1.0|0.007524487491056687|
|    max|          433060.0|   3258.554552877009|
+-------+------------------+--------------------+

    cvDataStd.describe("rating", "std_rating").show()
+-------+------------------+--------------------+
|summary|            rating|          std_rating|
+-------+------------------+--------------------+
|  count|           2430182|             2430182|
|   mean|15.229228921949055| 0.11459214252366869|
| stddev|  96.8352348860034|  0.7286355135902447|
|    min|               1.0|0.007524487491056687|
|    max|          101076.0|   760.5450976460457|
+-------+------------------+--------------------+

    testDataStd.describe("rating", "std_rating").show()
+-------+------------------+--------------------+
|summary|            rating|          std_rating|
+-------+------------------+--------------------+
|  count|           2428420|             2428420|
|   mean|15.472853130842276| 0.11642528983608988|
| stddev| 293.7031816022673|  2.2099659161110994|
|    min|               1.0|0.007524487491056687|
|    max|          439771.0|  3309.0513884294905|
+-------+------------------+--------------------+
     */
    // pipeline for ALS model
    val als = new ALS().
      setUserCol("uid").
      setItemCol("pid").
      setRatingCol("rating").
      setPredictionCol("prediction")

    val evaluator = new RegressionEvaluator().
      setMetricName("rmse").
      setLabelCol(als.getRatingCol).
      setPredictionCol(als.getPredictionCol)
    val paramGrid = new ParamGridBuilder().
      addGrid(als.rank, Array(1, 20, 40)).
      addGrid(als.maxIter, Array(5, 20, 40)).
      addGrid(als.regParam, Array(0.1, 1.0, 2.0)).
      addGrid(als.alpha, Array(1.0, 20.0, 40.0)).
      build
    val cv = new CrossValidator().
      setEstimator(als).
      setEstimatorParamMaps(paramGrid).
      setEvaluator(evaluator).
      setNumFolds(3)
    val cvModel = cv.fit(trainData)
    /*
    15/12/01 13:57:38 INFO storage.BlockManager: Removing RDD 40
    java.lang.IllegalArgumentException: requirement failed: Column prediction must be of type DoubleType but was actually FloatType.
            at scala.Predef$.require(Predef.scala:233)
            at org.apache.spark.ml.util.SchemaUtils$.checkColumnType(SchemaUtils.scala:42)
            at org.apache.spark.ml.evaluation.RegressionEvaluator.evaluate(RegressionEvaluator.scala:67)
            at org.apache.spark.ml.tuning.CrossValidator$$anonfun$fit$1.apply(CrossValidator.scala:94)
     */
    // training model
    val modelsCv = for(alpha <- Array(1.0, 20.0, 40.0);
      iter <- Array(5, 20, 40);
      rank <- Array(1, 20, 40);
      regParam <- Array(0.1, 1.0, 2.0))
      yield {
        val paramMap = ParamMap(als.alpha -> alpha).
          put(als.maxIter -> iter).
          put(als.rank -> rank).
          put(als.regParam -> regParam)
        val model = als.fit(trainData, paramMap)
        val predictions = model.
          transform(cvData). // : DataFrame = [uid: int, pid: int, rating: double, prediction: float]
          filter(!$"prediction".isNaN && !$"prediction".isNull). // predictions.filter($"prediction".isNaN || $"prediction".isNull).count() = 4001
          map{ case Row(uid: Int, pid: Int, rating: Double, prediction: Float) =>
            (rating, prediction.toDouble) }
        val rmse = computeRmse(predictions)
        (model, rmse, rank, alpha, iter, regParam)
      } // : Array[(org.apache.spark.ml.recommendation.ALSModel, Double, Int, Double, Int, Double)]
    /*
Resources: Driver[Xeon X5675@3.07GHz * 16 cores, 32GB]
Submitted 2015/12/03 11:19:47 ~ 2015/12/03 14:55:46, 215 mins

    modelsCv.size = 81
    modelsCv.
      sortBy{ case (model, rmse, rank, alpha, iter, regParam) => rmse }.
      foreach{ case (model, rmse, rank, alpha, iter, regParam) =>
        println(f"${rmse}%9.9s,${alpha}%4.4s,${iter}%2.2s,${rank}%2.2s,${regParam}%3.3s") }
102.89456, 1.0,20,40,2.0 ==> rank =40, reg = 2.0
102.89456,20.0,20,40,2.0
102.89456,40.0,20,40,2.0
103.01164, 1.0,40,40,2.0
103.01164,20.0,40,40,2.0
103.01164,40.0,40,40,2.0
105.93227,40.0, 5,40,2.0
105.93227, 1.0, 5,40,2.0
105.93227,20.0, 5,40,2.0
108.38374, 1.0,40,20,2.0 ==> rank =20, reg = 2.0
108.38374,20.0,40,20,2.0
108.38374,40.0,40,20,2.0
109.09482, 1.0,20,20,2.0
109.09482,40.0,20,20,2.0
109.09482,20.0,20,20,2.0
110.44896, 1.0, 5,20,2.0
110.44896,20.0, 5,20,2.0
110.44896,40.0, 5,20,2.0
110.58726,20.0, 5,40,1.0 ==> rank =40, reg = 1.0
110.58726,40.0, 5,40,1.0
110.58726, 1.0, 5,40,1.0
110.63332, 1.0,40,40,1.0
110.63332,20.0,40,40,1.0
110.63332,40.0,40,40,1.0
111.56681, 1.0,20,40,1.0
111.56681,20.0,20,40,1.0
111.56681,40.0,20,40,1.0
112.00155,20.0, 5, 1,1.0 ==> rank =1, reg = 1.0
112.00155, 1.0, 5, 1,1.0
112.00155,40.0, 5, 1,1.0
112.79100, 1.0, 5, 1,2.0 ==> rank =1, reg = 2.0
112.79100,20.0, 5, 1,2.0
112.79100,40.0, 5, 1,2.0
114.86664,40.0, 5,20,1.0 ==> rank =20, reg = 1.0
114.86664, 1.0, 5,20,1.0
114.86664,20.0, 5,20,1.0
115.62820, 1.0,40, 1,2.0 ==> rank =1, reg = 2.0
115.62820,20.0,40, 1,2.0
115.62820,40.0,40, 1,2.0
117.21149, 1.0,20, 1,2.0
117.21149,20.0,20, 1,2.0
117.21149,40.0,20, 1,2.0
117.59952, 1.0,20,20,1.0 ==> rank =20, reg = 1.0
117.59952,20.0,20,20,1.0
117.59952,40.0,20,20,1.0
118.15738, 1.0, 5, 1,0.1
118.15738,20.0, 5, 1,0.1
118.15738,40.0, 5, 1,0.1
118.62548, 1.0,40,20,1.0
118.62548,20.0,40,20,1.0
118.62548,40.0,40,20,1.0
119.14594, 1.0,40, 1,1.0
119.14594,20.0,40, 1,1.0
119.14594,40.0,40, 1,1.0
120.35391,40.0,20, 1,1.0
120.35391, 1.0,20, 1,1.0
120.35391,20.0,20, 1,1.0
140.81707, 1.0, 5,20,0.1
140.81707,20.0, 5,20,0.1
140.81707,40.0, 5,20,0.1
145.03270,40.0,20,40,0.1
145.03270, 1.0,20,40,0.1
145.03270,20.0,20,40,0.1
145.12334,40.0, 5,40,0.1
145.12334, 1.0, 5,40,0.1
145.12334,20.0, 5,40,0.1
147.96920,40.0,40,40,0.1
147.96920, 1.0,40,40,0.1
147.96920,20.0,40,40,0.1
153.66752, 1.0,20,20,0.1
153.66752,20.0,20,20,0.1
153.66752,40.0,20,20,0.1
164.82630, 1.0,40,20,0.1
164.82630,20.0,40,20,0.1
164.82630,40.0,40,20,0.1
230.97341, 1.0,20, 1,0.1
230.97341,20.0,20, 1,0.1
230.97341,40.0,20, 1,0.1
273.62274,40.0,40, 1,0.1
273.62274, 1.0,40, 1,0.1
273.62274,20.0,40, 1,0.1
     */
    // testing model
    val modelsTest = modelsCv.
        map{ case (model, rmseCv, rank, alpha, iter, regParam) =>
          val predictions = model.
            transform(testData).
            filter(!$"prediction".isNaN && !$"prediction".isNull).
            map{ case Row(uid: Int, pid: Int, rating: Double, prediction: Float) =>
              (rating, prediction.toDouble) }
          val rmseTest = computeRmse(predictions)
          (model, rmseCv, rmseTest, rank, alpha, iter, regParam) }
    /*
    modelsTest.
      sortBy{ case (model, rmseCv, rmseTest, rank, alpha, iter, regParam) => rmseCv }.
      zipWithIndex.
      sortBy{ case ((model, rmseCv, rmseTest, rank, alpha, iter, regParam), idxCv) => rmseTest }.
      zipWithIndex.
      foreach{ case (((model, rmseCv, rmseTest, rank, alpha, iter, regParam), idxCv), idxTest) =>
      println(f"${idxTest}%5s,${idxCv}%5s,${rmseTest}%9.9s,${rmseCv}%9.9s,${alpha}%4.4s,${iter}%2.2s,${rank}%2.2s,${regParam}%3.3s") }

    0,    9,298.59934,108.38374, 1.0,40,20,2.0
    1,   10,298.59934,108.38374,20.0,40,20,2.0
    2,   11,298.59934,108.38374,40.0,40,20,2.0
    3,   48,299.02734,118.62548, 1.0,40,20,1.0
    4,   49,299.02734,118.62548,20.0,40,20,1.0
    5,   50,299.02734,118.62548,40.0,40,20,1.0
    6,   12,299.05453,109.09482, 1.0,20,20,2.0
    7,   13,299.05453,109.09482,40.0,20,20,2.0
    8,   14,299.05453,109.09482,20.0,20,20,2.0
    9,   42,299.85345,117.59952, 1.0,20,20,1.0
   10,   43,299.85345,117.59952,20.0,20,20,1.0
   11,   44,299.85345,117.59952,40.0,20,20,1.0
   12,    3,300.91341,103.01164, 1.0,40,40,2.0
   13,    5,300.91341,103.01164,40.0,40,40,2.0
   14,    4,300.91341,103.01164,20.0,40,40,2.0
   15,    1,301.45575,102.89456,20.0,20,40,2.0
   16,    0,301.45575,102.89456, 1.0,20,40,2.0
   17,    2,301.45575,102.89456,40.0,20,40,2.0
   18,   33,301.88463,114.86664,40.0, 5,20,1.0
   19,   34,301.88463,114.86664, 1.0, 5,20,1.0
   20,   35,301.88463,114.86664,20.0, 5,20,1.0
   21,   15,301.97911,110.44896, 1.0, 5,20,2.0
   22,   16,301.97911,110.44896,20.0, 5,20,2.0
   23,   17,301.97911,110.44896,40.0, 5,20,2.0
   24,    6,303.01992,105.93227,40.0, 5,40,2.0
   25,    7,303.01992,105.93227, 1.0, 5,40,2.0
   26,    8,303.01992,105.93227,20.0, 5,40,2.0
   27,   21,303.20191,110.63332, 1.0,40,40,1.0
   28,   22,303.20191,110.63332,20.0,40,40,1.0
   29,   23,303.20191,110.63332,40.0,40,40,1.0
   30,   18,303.38122,110.58726,20.0, 5,40,1.0
   31,   19,303.38122,110.58726,40.0, 5,40,1.0
   32,   20,303.38122,110.58726, 1.0, 5,40,1.0
   33,   25,303.70015,111.56681,20.0,20,40,1.0
   34,   26,303.70015,111.56681,40.0,20,40,1.0
   35,   24,303.70015,111.56681, 1.0,20,40,1.0
   36,   30,306.96807,112.79100, 1.0, 5, 1,2.0
   37,   31,306.96807,112.79100,20.0, 5, 1,2.0
   38,   32,306.96807,112.79100,40.0, 5, 1,2.0
   39,   39,310.18995,117.21149, 1.0,20, 1,2.0
   40,   40,310.18995,117.21149,20.0,20, 1,2.0
   41,   41,310.18995,117.21149,40.0,20, 1,2.0
   42,   28,310.31866,112.00155, 1.0, 5, 1,1.0
   43,   27,310.31866,112.00155,20.0, 5, 1,1.0
   44,   29,310.31866,112.00155,40.0, 5, 1,1.0
   45,   36,310.49116,115.62820, 1.0,40, 1,2.0
   46,   37,310.49116,115.62820,20.0,40, 1,2.0
   47,   38,310.49116,115.62820,40.0,40, 1,2.0
   48,   63,310.59850,145.12334,40.0, 5,40,0.1
   49,   64,310.59850,145.12334, 1.0, 5,40,0.1
   50,   65,310.59850,145.12334,20.0, 5,40,0.1
   51,   54,313.49840,120.35391,40.0,20, 1,1.0
   52,   55,313.49840,120.35391, 1.0,20, 1,1.0
   53,   56,313.49840,120.35391,20.0,20, 1,1.0
   54,   51,314.11878,119.14594, 1.0,40, 1,1.0
   55,   52,314.11878,119.14594,20.0,40, 1,1.0
   56,   53,314.11878,119.14594,40.0,40, 1,1.0
   57,   60,315.56935,145.03270,40.0,20,40,0.1
   58,   61,315.56935,145.03270, 1.0,20,40,0.1
   59,   62,315.56935,145.03270,20.0,20,40,0.1
   60,   57,317.68960,140.81707, 1.0, 5,20,0.1
   61,   58,317.68960,140.81707,20.0, 5,20,0.1
   62,   59,317.68960,140.81707,40.0, 5,20,0.1
   63,   66,318.80183,147.96920,40.0,40,40,0.1
   64,   67,318.80183,147.96920, 1.0,40,40,0.1
   65,   68,318.80183,147.96920,20.0,40,40,0.1
   66,   45,323.38816,118.15738, 1.0, 5, 1,0.1
   67,   46,323.38816,118.15738,20.0, 5, 1,0.1
   68,   47,323.38816,118.15738,40.0, 5, 1,0.1
   69,   69,324.98540,153.66752, 1.0,20,20,0.1
   70,   70,324.98540,153.66752,20.0,20,20,0.1
   71,   71,324.98540,153.66752,40.0,20,20,0.1
   72,   72,327.69184,164.82630, 1.0,40,20,0.1
   73,   73,327.69184,164.82630,20.0,40,20,0.1
   74,   74,327.69184,164.82630,40.0,40,20,0.1
   75,   75,334.80691,230.97341, 1.0,20, 1,0.1
   76,   76,334.80691,230.97341,20.0,20, 1,0.1
   77,   77,334.80691,230.97341,40.0,20, 1,0.1
   78,   78,347.82040,273.62274,40.0,40, 1,0.1
   79,   80,347.82040,273.62274,20.0,40, 1,0.1
   80,   79,347.82040,273.62274, 1.0,40, 1,0.1
     */
    val bestModel = modelsCv.sortBy{case (model, rmse, rank, alpha, iter, regParam) => rmse}.head._1
    // : (org.apache.spark.ml.recommendation.ALSModel, Double, Int, Double, Int, Double) = (als_3fccd0a7b397,44.30403266876076,20,10.0,20,1.0)
    val predictTrain = bestModel.transform(testData) // : DataFrame = [uid: int, pid: int, rating: double, prediction: float]
    /*
predictTrain.show(5)
+-------+---+------+----------+
|    uid|pid|rating|prediction|
+-------+---+------+----------+
|1000231|231|    13|  20.28318|
|1001031|231|    17| 12.261612|
|1002031|231|     3| 3.2966247|
|1002231|231|    13|  9.554958|
|1006831|231|    14| 10.697116|
+-------+---+------+----------+

predictCv.show(5)
+-------+---+------+----------+
|    uid|pid|rating|prediction|
+-------+---+------+----------+
|1044831|231|    31|  9.110821|
|1054231|231|     8|  43.59666|
|2004631|231|     6| 5.7561383|
|2128031|231|    10| 7.4704113|
|2146231|231|    15|  32.40709|
+-------+---+------+----------+
     */

  }

  def model2 = {
    val artistAlias = parseArtAli()
    val userPlayList = parseUserPlay()
    val artistData = parseArtName()
    // preparing data
    val userRating = userPlayList.
      join(artistAlias, userPlayList("usrAliasId") === artistAlias("artAliasId"), "left_outer").
      selectExpr("cast(userId as int) as uid","cast(nvl(artistId, usrAliasId) as int) as pid ","cast(count as double) as rating")
      // DataFrame = [uid: int, pid: int, rating: double]
    val Array(trainData, cvData, testData) = userRating.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache() // cvData.count() = 2429720
    testData.cache()
    // pipeline for ALS model
    val als = new ALS().
      setUserCol("uid").
      setItemCol("pid").
      setRatingCol("rating").
      setPredictionCol("prediction")
    // training model
    val modelsCv = for(alpha <- Array(1.0, 10);
                       iter <- Array(20);
                       rank <- Array(20, 40);
                       regParam <- Array(2.0, 4.0, 8.0))
    yield {
      val paramMap = ParamMap(als.alpha -> alpha).
        put(als.maxIter -> iter).
        put(als.rank -> rank).
        put(als.regParam -> regParam)
      val model = als.fit(trainData, paramMap)
      val predictions = model.
        transform(cvData).  // : DataFrame = [uid: int, pid: int, rating: double, prediction: float]
        filter(!$"prediction".isNaN && !$"prediction".isNull). // predictions.filter($"prediction".isNaN || $"prediction".isNull).count() = 4001
        map{ case Row(uid: Int, pid: Int, rating: Double, prediction: Float) =>
        (rating, prediction.toDouble) }
      val rmse = computeRmse(predictions)
      (model, rmse, rank, alpha, iter, regParam)
    } // : Array[(org.apache.spark.ml.recommendation.ALSModel, Double, Int, Double, Int, Double)]
    /*
Resources: Driver[Xeon X5675@3.07GHz * 16 cores, 32GB]
Submitted 2015/12/09 16:11:11 ~ 2015/12/09 16:43:55, 32 mins

    modelsCv.size = 12
    modelsCv.
      sortBy{ case (model, rmse, rank, alpha, iter, regParam) => rmse }.
      foreach{ case (model, rmse, rank, alpha, iter, regParam) =>
      println(f"${rmse}%9.9s,${alpha}%4.4s,${iter}%2.2s,${rank}%3.3s,${regParam}%3.3s") }

74.281254, 1.0,20, 40,8.0
74.281254,10.0,20, 40,8.0
76.109640, 1.0,20, 20,8.0
76.109640,10.0,20, 20,8.0
78.907838, 1.0,20, 40,4.0
78.907838,10.0,20, 40,4.0
83.918632, 1.0,20, 20,4.0
83.918632,10.0,20, 20,4.0
90.304407,10.0,20, 40,2.0
90.304407, 1.0,20, 40,2.0
91.994248, 1.0,20, 20,2.0
91.994248,10.0,20, 20,2.0
     */
    // testing model
    val modelsTest = modelsCv.
      map{ case (model, rmseCv, rank, alpha, iter, regParam) =>
      val predictions = model.
        transform(testData).
        filter(!$"prediction".isNaN && !$"prediction".isNull).
        map{ case Row(uid: Int, pid: Int, rating: Double, prediction: Float) =>
        (rating, prediction.toDouble) }
      val rmseTest = computeRmse(predictions)
      (model, rmseCv, rmseTest, rank, alpha, iter, regParam) }

    /*
Submitted 2015/12/09 16:46:51 ~ 2015/12/09 16:48:37, 2 mins

    modelsTest.
      sortBy{ case (model, rmseCv, rmseTest, rank, alpha, iter, regParam) => rmseCv }.
      zipWithIndex.
      sortBy{ case ((model, rmseCv, rmseTest, rank, alpha, iter, regParam), idxCv) => rmseTest }.
      zipWithIndex.
      foreach{ case (((model, rmseCv, rmseTest, rank, alpha, iter, regParam), idxCv), idxTest) =>
      println(f"${idxTest}%2s,${idxCv}%2s,${rmseTest}%9.9s,${rmseCv}%9.9s,${alpha}%4.4s,${iter}%2.2s,${rank}%3.3s,${regParam}%3.3s") }

 0, 0,83.698045,74.281254, 1.0,20, 40,8.0
 1, 1,83.698045,74.281254,10.0,20, 40,8.0
 2, 3,85.655462,76.109640,10.0,20, 20,8.0
 3, 2,85.655462,76.109640, 1.0,20, 20,8.0
 4, 4,87.739403,78.907838, 1.0,20, 40,4.0
 5, 5,87.739403,78.907838,10.0,20, 40,4.0
 6, 8,94.220677,90.304407,10.0,20, 40,2.0
 7, 9,94.220677,90.304407, 1.0,20, 40,2.0
 8, 6,95.795563,83.918632, 1.0,20, 20,4.0
 9, 7,95.795563,83.918632,10.0,20, 20,4.0
10,10,109.93892,91.994248, 1.0,20, 20,2.0
11,11,109.93892,91.994248,10.0,20, 20,2.0
     */
  }
}
