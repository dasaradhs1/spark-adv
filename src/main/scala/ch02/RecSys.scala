package ch02

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.rdd._
import org.apache.spark.mllib.fpm._
/**
 * Created by leorick on 2016/1/20.
 */
object RecSys {

  def wifiUsrSpot(sc:SparkContext) = {
    import
    val src = sc.textFile("file:///home/leoricklin/dataset/wifiusrspot/20151101")
    /*
    src.take(5)
= Array(0932965782,PH014348, 0921351536,PH002579, 0978608915,PH010660,PH107010,PH107649,PH105066, 0921798090,PH083558,PH083054,PH018410,PH083058,PH083058,PH083046,PH083749,PH083557, 0926070151,PH050277,PH105090,PH090692,PH007602)
     */
    val usrspotscnt = src.map{ line => line.split("""\p{Punct}""").map{_.trim}.tail.size }
    val ret = usrspotscnt.map{_.toDouble}.histogram(10)
    val out1 = ret._1.zip(ret._2).map{ case (avgconn, usrcnt) => f"1\t${avgconn}\t${usrcnt}"}.mkString("\n")
    val out2 = ret._1.zip(ret._2).map{ case (avgconn, usrcnt) => f"2\t${avgconn}\t${usrcnt+50}"}.mkString("\n")
    print(s"%table\navg. conn times\tuser count\n${out1}")
    print(s"""%table
              type\tavg. conn times\tuser count
              ${out1}
              ${out2}""")

    val spots = src.map{ line => line.split("""\p{Punct}""").map{_.trim}.tail }
    val uniqspots = spots.map{ itemset => itemset.distinct}
    uniqspots.cache()
    val txCnt = uniqspots.count
    /*
    spots.count = 551591
    spots.take(5)
= Array(Array(PH014348), Array(PH002579), Array(PH010660, PH107010, PH107649, PH105066), Array(PH083558, PH083054, PH018410, PH083058, PH083058, PH083046, PH083749, PH083557), Array(PH050277, PH105090, PH090692, PH007602))
     */
    val fpg = new FPGrowth().setMinSupport(0.001)
    val model: FPGrowthModel[String] = fpg.run(uniqspots)
    model.freqItemsets.collect().
      sortBy{ itemset => itemset.items.sortBy((item: String) => item).head }.
      foreach { itemset => println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq) }
    /*
[PH037815], 1610, P=1610/551591= 0.00292, 台北捷運-台北車站B3淡水線(公話1210121)
[PH083558], 2146, P=2146/551591= 0.00389, 台北捷運-台北車站公話1226705
[PH082463], 1335, P=1335/551591= 0.00242, 公話-捷運江子翠站B2樓月台1960030
[PH082465], 1299, P=1299/551591= 0.00236, 公話-捷運新埔站B2樓月台北端1966441
[PH084377], 1207, P=1207/551591= 0.00219, 台北捷運忠孝敦化站公話1261450
[PH084921], 1187, P=1187/551591= 0.00215, 台北捷運忠孝復興站公話1261446
[PH037815,PH083558], 584, P=584/551591=0.00106, lift = 0.00106/(0.00292*0.00389) = 93.31972
[PH082465,PH082463], 750, P=750/551591=0.00136, lift = 0.00136/(0.00236*0.00242) = 238.1286
[PH084921,PH084377], 562, P=562/551591=0.00102, lift = 0.00102/(0.00215*0.00219) = 216.6295
     */
    val modelwlift = getFpModelLift(sc, txCnt, model)
  }

  def getFpModelLift(sc:SparkContext, txCnt:Long, fpModel:FPGrowthModel[String])
  : RDD[(Double, (Array[String], Long, Long, Double, Double, Double))] = {
    // 取得單一item的出現機率
    val fpItMap = fpModel.freqItemsets.filter{
      itset => itset.items.length == 1
    }.map{
      itset => (itset.items(0), itset.freq.toDouble/txCnt.toDouble)
    }.collectAsMap()
    val fpItBc = sc.broadcast(fpItMap)
    // 取得freqitemset的lift
    val fpmodelwLift = fpModel.freqItemsets.mapPartitions{ ite =>
      val fpItMap = fpItBc.value
      ite.filter{
        itset => itset.items.length > 1
      }.map{ itset =>
        val indProb = itset.items.map{
          item => fpItMap.getOrElse(item, 1.0D)
        }.toList.reduce( (a,b) => a * b )
        val itsetProb = itset.freq.toDouble/txCnt.toDouble
        ( itset.items, itset.freq, txCnt, indProb, itsetProb, itsetProb/indProb )
      }
      // 基於lift值排序
    }.map{
      case ( items, freq, txCnt, indProb, itsetProb, lift ) => (lift, ( items, freq, txCnt, indProb, itsetProb, lift ))
    }.sortByKey(false)
    fpmodelwLift
  }

  def main(args: Array[String]) {
    val appName = "Wifi Analysis"
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)

  }
}
