package ch03

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation._
import org.apache.spark.mllib.util._

/**
 * Created by leorick on 2015/9/2.
 */
object RecBasicALS extends Serializable {
  val appName = "Basic Decision Tree for Spam classification"
  val sparkConf = new SparkConf().setAppName(appName)
  val sc = new SparkContext(sparkConf)
  def model1 = {
    // val rawArt = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/artist_data.txt")
    val rawArt = sc.textFile("file:///media/sf_WORKSPACE.2W/dataset/AAS/audioscrobbler/artist_data.txt")
    val artByID : RDD[(Int, String)] = rawArt.flatMap { line =>
      val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      } else {
        try {
          Some((id.toInt, name.trim))
        } catch {
          case e: NumberFormatException => None
        }
      }
    }
    /*
    rawArt.count : Long = 1848707
    rawArt.take(2) : Array[String] = Array(1134999     06Crazy Life, 6821360   Pang Nakarin)
    artByID.first  : (Int, String) = (1134999,06Crazy Life)
    artByID.lookup(6803336) : Seq[String] = WrappedArray(Aerosmith (unplugged))
     */
    // val rawArtAli = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/artist_alias.txt")
    val rawArtAli = sc.textFile("file:///media/sf_WORKSPACE.2W/dataset/AAS/audioscrobbler/artist_alias.txt")
    val artAli : scala.collection.Map[Int,Int] = rawArtAli.flatMap { line =>
      val tokens = line.split('\t')
      if (tokens(0).isEmpty) {
        None
      } else {
        Some((tokens(0).toInt, tokens(1).toInt))
      }
    }.collectAsMap()
    val bArtistAlias = sc.broadcast(artAli)
    /*
    rawArtAli.take(2) : Array[String] = Array(1092764     1000311, 1095122        1000557)
    artAli.head : (Int, Int) = (6803336,1000010)
     */
    // val rawUA = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/user_artist_data.txt")
    val rawUA = sc.textFile("file:///media/sf_WORKSPACE.2W/dataset/AAS/audioscrobbler/user_artist_data.txt")
    val allData : RDD[Rating] = rawUA.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)     // tokenize
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)  // clean data: map to unique artist ID
      Rating(userID, finalArtistID, count)
    }.cache()

    val model: MatrixFactorizationModel = ALS.trainImplicit(allData, 10, 5, 0.01, 1.0)
    /*
    rawUA.take(2) : Array[String] = Array(1000002 1 55, 1000002 1000006 33)
    :load /media/sf_WORKSPACE.2W/spark-util/src/main/scala/tw/com/chttl/spark/mllib/util/NAStatCounter.scala
    :load /media/sf_WORKSPACE.2W/spark-util/src/main/scala/tw/com/chttl/spark/mllib/util/NAStat.scala
    import tw.com.chttl.spark.mllib.util._
    NAStat.statsWithMissing(rawUA.map{ line => line.split(' ').map(tok => tok.toDouble) })
     = Array(
      (uid)    stats: (count: 24,296,858, mean: 1947573.265353, stdev: 496000.544975, max: 2443548.000000, min: 90.000000), NaN: 0
     ,(pid)    stats: (count: 24,296,858, mean: 1718704.093757, stdev: 2539389.040171, max: 10794401.000000, min: 1.000000), NaN: 0
     ,(rating) stats: (count: 24,296,858, mean: 15.295762, stdev: 153.915321, max: 439771.000000, min: 1.000000), NaN: 0)

    allData.zipWithIndex().filter{ case (rating, idx) => idx >= 5959 && idx < 5969}.collect()
15/09/04 15:21:59 INFO DAGScheduler: Job 27 finished: collect at <console>:37, took 0.718516 s
res9: Array[(org.apache.spark.mllib.recommendation.Rating, Long)] = Array((Rating(1000035,599,1.0),5959), (Rating(1000035,606,5.0),5960), (Rating(1000035,676,1.0),5961), (Rating(1000035,699,2.0),5962), (Rating(1000035,2003588,10.0),5963), (Rating(1000035,721,6.0),5964), (Rating(1000035,722,1.0),5965), (Rating(1000035,740,2.0),5966), (Rating(1000035,744,2.0),5967), (Rating(1000035,78,8.0),5968))

    allData.zipWithUniqueId().filter{ case (rating, idx) => idx >= 5959 && idx < 5969 }.collect()
15/09/04 15:21:26 INFO DAGScheduler: Job 25 finished: collect at <console>:37, took 0.782116 s
res8: Array[(org.apache.spark.mllib.recommendation.Rating, Long)] = Array((Rating(1000002,1016357,1.0),5967), (Rating(1038228,2290019,1.0),5968), (Rating(2068548,1015700,13.0),5959), (Rating(2107287,530,23.0),5960), (Rating(2155794,1044497,52.0),5961), (Rating(2205354,1000052,10.0),5962), (Rating(2254599,1003681,1.0),5963), (Rating(2300989,1000781,1.0),5964), (Rating(2344653,1198,80.0),5965), (Rating(2399869,1143074,3.0),5966))

    # Host=10.176.32.76 with 16g, 16 cores => less than 2 mins
    model.userFeatures.mapValues(_.mkString(", ")).first()
     */
    //
    val art4U = allData.filter { rating => rating.user  == 2093760
    }.map { rating => rating.product }.collect().toSet
    /*
    = Set(1255340, 942, 1180, 813, 378)
     */
    artByID.filter { case (id, name) => art4U.contains(id) }.values.collect().foreach(println)
    /*
David Gray
Blackalicious
Jurassic 5
The Saw Doctors
Xzibit
     */
    val rec4U = model.recommendProducts(2093760, 5)
    val recArt4U = rec4U.map(_.product).toSet
    artByID.filter { case (id, name) => recArt4U.contains(id) }.values.collect().foreach(println)
/*
50 Cent
Snoop Dogg
Jay-Z
2Pac
The Game
 */
    val predictData = model.predict(allData.map(r => (r.user, r.product))).groupBy(r => (r.user,r.product))
    val cvData: RDD[((Int, Int), (Iterable[Rating], Iterable[Rating]))] = allData.groupBy(r => (r.user,r.product)).join(predictData)
    val ret2: Array[((Int, Int), (Iterable[Rating], Iterable[Rating]))] = cvData.take(5)
    ret2.map{ case ((uid,pid), (ite1, ite2)) =>
      val rating1 = ite1    .map(r=> f"${r.user},${r.product},${r.rating}").mkString("[","|","]")
      val rating2 = ite2.map(r=> f"${r.user},${r.product},${r.rating}").mkString("[","|","]")
      f"${uid},${pid},${rating1},${rating2}"
    }
/*
    = Array(
      1049114,1044731,[1049114,1044731, 1.0],[1049114,1044731,0.05502531301749228]
    , 2069658,1026826,[2069658,1026826,17.0],[2069658,1026826,0.2585166201947547]
    , 2277229,1005990,[2277229,1005990, 2.0],[2277229,1005990,0.9733582835509411]
    , 1071243,   4475,[1071243,   4475, 2.0],[1071243,   4475,0.9920412839840766]
    , 2086203,1007719,[2086203,1007719, 3.0],[2086203,1007719,0.13567941441949843])
 */


    val ret1 = allData.take(3)
    ret1.map{ r => (r, model.predict(r.user, r.product))
    }.foreach { case (r, predict) =>
      println(f"uid=${r.user},pid=${r.product},rating=${r.rating},predect=${predict}")
    }
    /*
    udi=1000002,pid=1      ,rating=55.0,predect=0.8052675069971611
    udi=1000002,pid=1000006,rating=33.0,predect=0.01459615004443604
    udi=1000002,pid=1000007,rating=8.0 ,predect=0.018664514737406563
     */
  }

  def model2 = {
    /*
    :load /media/leo/1000_D/WORKSPACE.2/spark-adv/src/main/scala/ch03/RecUtil.scala
    :load /home/leoricklin/dataset/RecUtil.scala
     */
    // read artist alias and load as bv.
    /*
    val rawArtAli = sc.textFile("file:///media/leo/1000_D/WORKSPACE.2/dataset/AAS/audioscrobbler/artist_alias.txt")
     */
    val rawArtAli = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/artist_alias.txt")
    val bArtAli: Broadcast[scala.collection.Map[Int, Int]] = sc.broadcast(RecUtil.buildArtistAlias(rawArtAli))
    // prepare train / cv dataset
    /*
    val rawUA = sc.textFile("file:///D:/WORK/profiledata/user_artist_data.txt")
    val rawUA = sc.textFile("file:///media/leo/1000_D/WORKSPACE.2/dataset/AAS/audioscrobbler/user_artist_data.txt")
     */
    val rawUA = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/user_artist_data.txt")
    val allData: RDD[Rating] = RecUtil.buildRatings(rawUA , bArtAli)
    val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
    trainData.cache();cvData.cache()
    /*
    allData.map{r=>r.user}.distinct().count() = 148,111
    allData.map{r=>r.product}.distinct().count() = 1,568,126
     */
    // prepare complete distinct artist IDs and load as bv.
    val bAllArtIDs: Broadcast[Array[Int]] = sc.broadcast( allData.map(_.product).distinct().collect() )
    /*
    bAllArtIDs.value.size = 1568126
    by 用戶計算撥放歌手的統計分佈
    trainData.count = 21867152
    trainData.map{ r => (r.user, r.product) }.groupByKey().map{ case (uid, ite) => ite.size.toDouble }.stats()
     : StatCounter = (count: 147741, mean: 148.005679, stdev: 196.162648, max: 6156.000000, min: 1.000000)
    by 用戶計算撥放歌手(不重複)的統計分佈
    trainData.map{ r => (r.user, r.product) }.groupByKey().map{ case (uid, ite) => ite.toSet.size.toDouble }.stats()
     : StatCounter = (count: 147741, mean: 147.095038, stdev: 194.512160, max: 6077.000000, min: 1.000000)
     */
    val rawArtData = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/artist_data.txt")
    val artByID : RDD[(Int, String)] = RecUtil.buildArtistByID(rawArtData)
    /*
    val model: MatrixFactorizationModel = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
    val auc = RecUtil.areaUnderCurve( cvData, bAllArtIDs, model.predict )
    = 0.9657771805115158
     */
    val evaluations: Array[((Int, Double, Double), Double, MatrixFactorizationModel)] =
      for (rank <- Array(10, 50); lambda <- Array(1.0, 0.0001); alpha <- Array(1.0, 40.0)
      ) yield {
        val model = ALS.trainImplicit(trainData, rank, 10, lambda, alpha)
        val auc = RecUtil.areaUnderCurve(cvData, bAllArtIDs, model.predict)
        ((rank, lambda, alpha), auc, model)
      }
    /* Host=10.176.32.76 with 16g, 16 cores => it takes about 20 mins
    evaluations.sortBy(_._2).reverse.foreach(println)
((50,1.0   ,40.0),0.9774516176787654)
((10,1.0   ,40.0),0.9767085999291152)
((10,1.0E-4,40.0),0.9765836016288412)
((50,1.0E-4,40.0),0.9764744530287514)
((10,1.0   ,1.0) ,0.969072856078996)
((50,1.0   ,1.0) ,0.9669777643405537)
((10,1.0E-4,1.0) ,0.9645617800564434)
((50,1.0E-4,1.0) ,0.954274147046807)
     */

    evaluations.zipWithIndex.sortBy{
      case (((rank, lambda, alpha), auc, model), idx) => auc
    }.reverse.foreach{ case (((rank, lambda, alpha), auc, model), idx) =>
      println(f"${idx}%2s: rank=${rank}%2s, lambda=${lambda}%7s, alpha=${alpha}%2s, auc=${auc}%1.20f")
    }
    /*
 5: rank=50, lambda=    1.0, alpha=40.0, auc=0.97736964527017670000
 1: rank=10, lambda=    1.0, alpha=40.0, auc=0.97669645023407810000
 7: rank=50, lambda= 1.0E-4, alpha=40.0, auc=0.97634219213662010000
 3: rank=10, lambda= 1.0E-4, alpha=40.0, auc=0.97617552827304570000
 0: rank=10, lambda=    1.0, alpha=1.0, auc=0.96819068316025790000
 4: rank=50, lambda=    1.0, alpha=1.0, auc=0.96696343437638230000
 2: rank=10, lambda= 1.0E-4, alpha=1.0, auc=0.96427820962852550000
 6: rank=50, lambda= 1.0E-4, alpha=1.0, auc=0.95454089107662110000

    evaluations(5) match {
      case ((rank, lambda, alpha), auc, model) => println(f"rank=${rank}%2s, lambda=${lambda}%7s, alpha=${alpha}%2s, auc=${auc}%1.20f")
      case _ =>
    }
    rank=50, lambda=    1.0, alpha=40.0, auc=0.97736964527017670000

    val someUsers: Array[Int] = allData.map(_.user).distinct().take(100)
    val someRecommendations: Array[Array[Rating]] = someUsers.map(userID => model.recommendProducts(userID, 5))
     */
    val bestModel = evaluations(5)._3
    bestModel.save(sc, "file:///home/leoricklin/dataset/recmodel.20150908")
    val rec4U: Array[Rating] = bestModel.recommendProducts(2093760, 5)
    /*
    Array(
      Rating(2093760,1001819,0.3386221630016077)
    , Rating(2093760,2814,0.337765452795137)
    , Rating(2093760,1300642,0.33647273487588614)
    , Rating(2093760,1003249,0.3343293337548755)
    , Rating(2093760,4185,0.33324379677955784))

    val rec4U: Array[Rating] = bestModel.recommendProducts(2093760, 5)
     Array(
       Rating(2093760,1001819,0.3386221630016077)
     , Rating(2093760,2814,0.337765452795137)
     , Rating(2093760,1300642,0.33647273487588614)
     , Rating(2093760,1003249,0.3343293337548755)
     , Rating(2093760,4185,0.33324379677955784))
     */
    sc.parallelize(rec4U.map{r => (r.product, r)}).join(artByID
    ).map{case (pid, (rating, pname)) => pname}.collect()
    /*
    = Array(Ludacris, The Game, D12, 2Pac, 50 Cent)
     */
    val bRecArt4U = sc.broadcast(rec4U.map(r => r.product).toSet)
    val artName = artByID.mapPartitions{ite =>
      val recArt4U = bRecArt4U.value
      ite.map{ case (id, name) =>
        if (recArt4U.contains(id)) {
          Some(name)
        } else {
          None
        }
      }
    }
  }

  def model3 = {
    val rawArtData = sc.textFile("file:///home/leoricklin/dataset/audioscrobbler/artist_data.txt")
    val artByID : RDD[(Int, String)] = RecUtil.buildArtistByID(rawArtData)
    //
    val bestModel = MatrixFactorizationModel.load(sc, "file:///home/leoricklin/dataset/recmodel.20150908")
    val rec4U: Array[Rating] = bestModel.recommendProducts(2093760, 5)
    /*
    Array(
      Rating(2093760,1001819,0.3386221630016077)
    , Rating(2093760,2814,0.337765452795137)
    , Rating(2093760,1300642,0.33647273487588614)
    , Rating(2093760,1003249,0.3343293337548755)
    , Rating(2093760,4185,0.33324379677955784))
     */
    val recArt4U = rec4U.map(r => r.product).toSet
    artByID.filter{ case (id, name) =>
      recArt4U.contains(id)
    }.map{ case (id, name) => name }.collect()
    /*
    = Array(50 Cent, D12, Ludacris, 2Pac, The Game)
     */
  }
}
