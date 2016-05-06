package woplus

import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row,SQLContext}
import org.apache.spark.{SparkContext, SparkConf}
import scala.collection.immutable.IndexedSeq
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import tw.com.chttl.spark.mllib.util.NAStat

/**
 * Created by leorick on 2016/5/4.
 */
object Profile {
  val appName = "woplus.profile"

  case class userProfile(imei:String, features:Seq[Double])

  def userProfile2Str(userprofile:userProfile): String = {
    userprofile.imei + "," + userprofile.features.map{_.toString}.mkString(",")
  }

  def str2Userprofile(line:String): userProfile = {
    val ary = line.split(""",""")
    userProfile(ary(0),ary.tail.map{_.toDouble})
  }

  def row2Userprofile(row:Row): userProfile = {
    userProfile(row.getString(0), row.getSeq(1))
  }

  def bytes2hex(bytes: Array[Byte], sep: Option[String] = None): String = {
    sep match {
      case None => bytes.map("%02x".format(_)).mkString
      case _ => bytes.map("%02x".format(_)).mkString(sep.get)
    }
  }

  def loadProfileSrc(sc:SparkContext, path:String): RDD[String] = {
    sc.textFile(path)
  }

  def splitProfileSrc(src:RDD[String]): RDD[(String, Array[Double])] = {
    src.map{_.split(",")}.
      filter{ toks => !toks(0).contains("IMEI") }.  // remove header
      map{ toks => toks.map{ tok => tok.replaceAll(""""""", "") } }.
      map{ toks =>
        val buf = toks.toBuffer
        buf.remove(6,2)         // 是否有跨省行为,是否有出国行为
        val imei = buf.remove(0)
        buf.remove(buf.size-2)  // 访问其他类网站的次数
        (imei, buf.map{ tok => if (tok.size == 0) "0" else tok }.map{_.toDouble}.toArray) }
  }

  def saveUserProfiles(sc:SparkContext, rdd:RDD[userProfile], path:String) = {
    import java.io._
    val file = new File(path)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write( rdd.map{ userprofile => userProfile2Str(userprofile) }.collect().mkString("\n") )
    bw.close()
  }

  def loadUserProfiles(sc:SparkContext, path:String): RDD[userProfile] = {
    sc.textFile(path).map{ line => str2Userprofile(line)}
  }

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    // 載入 & 分隔資料
    var path = "file:///home/leoricklin/dataset/woplus/userprofile"
    val profileToks = splitProfileSrc( loadProfileSrc(sc, path) )
    // 探索欄位
    NAStat.statsWithMissing( profileToks.map{ case(imei, features) => Array(features.size.toDouble) } )
    /*
Array(stats: (count: 2391166, mean: 20.000000, stdev: 0.000000, max: 20.000000, min: 20.000000), NaN: 0)
     */
    /*
Array(stats: (count: 2391166, mean: 21.000000, stdev: 0.000000, max: 21.000000, min: 21.000000), NaN: 0)

    println( profileToks.take(5).
      map{ case(imei, features) =>
      f"[${imei}][${features.mkString(",")}]" }.
      mkString("\n") )
[2ab41bf4da442c8b2f3306c28d15d919][0.0,0.0,0.0,0.0,180.0,4.0,9.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0]
[d2aa3932a2e30174c48db0903c5d6be7][2.0,0.0,0.0,0.0,138.0,17.0,548.0,0.0,0.0,0.0,5.0,0.0,0.0,0.0,29.0,11.0,1.0,0.0,14.0,0.0,23.0]
[2d2ffd78336770d589a239943a3b8491][0.0,0.0,0.0,889.0,3.0,3288.0,1220.0,0.0,19.0,0.0,323.0,5.0,0.0,0.0,210.0,103.0,25.0,0.0,1.0,0.0,131.0]
[181271cc81d8e27efcf88dd8a67233b6][0.0,0.0,0.0,0.0,23.0,692.0,294.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.0,0.0,19.0,0.0,0.0,0.0,70.0]
[1e1528b7c57a948b0c3bd05eec973478][0.0,0.0,0.0,0.0,21.0,7192.0,334.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,102.0,0.0,0.0,0.0,0.0,0.0,548.0]
     */
    /*
    wc -l /media/sf_WORKSPACE.2W/dataset/woplus/userprofile/profile.csv
2391167 /media/sf_WORKSPACE.2W/dataset/woplus/userprofile/profile.csv

    profileToks.filter{ toks =>
        toks(0).contains("2ab41bf4da442c8b2f3306c28d15d919") ||
        toks(0).contains("d2aa3932a2e30174c48db0903c5d6be7") ||
        toks(0).contains("2d2ffd78336770d589a239943a3b8491") ||
        toks(0).contains("181271cc81d8e27efcf88dd8a67233b6") }.
      map{ toks => Array(toks(0),toks(5),toks(6),toks(7),toks(8)) }.
      collect().map{ toks => f"[${toks(0)}][${toks(1)}][${bytes2hex(toks(2).getBytes())}][${bytes2hex(toks(3).getBytes())}][${toks(4)}]"}.
      mkString("\n")
[2ab41bf4da442c8b2f3306c28d15d919][180][efbfbdefbfbd][efbfbdefbfbd][4]
[d2aa3932a2e30174c48db0903c5d6be7][138][efbfbdefbfbd][efbfbdefbfbd][17]
[2d2ffd78336770d589a239943a3b8491][3]  [efbfbdefbfbd][efbfbdefbfbd][3288]
[181271cc81d8e27efcf88dd8a67233b6][23] [efbfbdefbfbd][efbfbdefbfbd][692]

    profileToks.map{ toks => (bytes2hex(toks(6).getBytes()) , bytes2hex(toks(7).getBytes())) }.
      distinct().collect()
= Array((efbfbdefbfbd,efbfbdefbfbd))

    profileToks.take(5).
      map{ toks =>
        val buf = toks.toBuffer
        buf.remove(6,2)
        buf }.
      map{ toks => f"${toks.mkString(",")}"}.
      mkString("\n")
2ab41bf4da442c8b2f3306c28d15d919,0,0,0,0,180,4,9,0,3,0,0,0,0,0,3,0,0,0,0,0,0
d2aa3932a2e30174c48db0903c5d6be7,2,0,0,0,138,17,548,0,0,0,5,0,0,0,29,11,1,0,14,0,23
2d2ffd78336770d589a239943a3b8491,0,0,0,889,3,3288,1220,0,19,0,323,5,0,0,210,103,25,0,1,0,131
181271cc81d8e27efcf88dd8a67233b6,0,0,0,0,23,692,294,0,0,0,0,0,0,0,8,0,19,0,0,0,70
1e1528b7c57a948b0c3bd05eec973478,0,0,0,0,21,7192,334,0,0,0,1,0,0,0,102,0,0,0,0,0,548
     */
    NAStat.statsWithMissing(profileToks.map{ case(imei, features) => features }).
      foreach(println)
    /*
00 stats: (count: 2391166, mean: 0.843520, stdev: 10.790137, max: 2500.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.091627, stdev: 13.801986, max: 6827.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 10.776034, stdev: 467.619190, max: 20304.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 40202.870580, stdev: 637951.569737, max: 18385272.000000, min: 0.000000), NaN: 0
04 stats: (count: 2391166, mean: 11.910784, stdev: 32.353184, max: 1622.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 889.027098, stdev: 7266.649560, max: 1678728.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 650.014548, stdev: 3198.212783, max: 2981364.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.274488, stdev: 10.085989, max: 5451.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 8.255610, stdev: 113.620084, max: 43870.000000, min: 0.000000), NaN: 0
09 stats: (count: 2391166, mean: 6.053055, stdev: 68.929221, max: 40668.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 296.038620, stdev: 3038.513531, max: 787784.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 32.829360, stdev: 407.369029, max: 151032.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 115.615995, stdev: 4261.692426, max: 656583.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 9.985474, stdev: 102.062806, max: 47678.000000, min: 0.000000), NaN: 0
14 stats: (count: 2391166, mean: 72.629991, stdev: 443.191626, max: 145343.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 20.437082, stdev: 804.888987, max: 136097.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 7.112892, stdev: 76.967212, max: 25587.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 4.875653, stdev: 78.436967, max: 8882.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 39.477453, stdev: 323.215951, max: 183722.000000, min: 0.000000), NaN: 0
19 stats: (count: 2391166, mean: 66.446228, stdev: 611.911873, max: 568889.000000, min: 0.000000), NaN: 0
     */
    /*
0 stats: (count: 2391166, mean: 0.843520, stdev: 10.790137, max: 2500.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.091627, stdev: 13.801986, max: 6827.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 10.776034, stdev: 467.619190, max: 20304.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 40202.870580, stdev: 637951.569737, max: 18385272.000000, min: 0.000000), NaN: 0
4 stats: (count: 2391166, mean: 11.910784, stdev: 32.353184, max: 1622.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 889.027098, stdev: 7266.649560, max: 1678728.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 650.014548, stdev: 3198.212783, max: 2981364.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.274488, stdev: 10.085989, max: 5451.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 8.255610, stdev: 113.620084, max: 43870.000000, min: 0.000000), NaN: 0
9 stats: (count: 2391166, mean: 6.053055, stdev: 68.929221, max: 40668.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 296.038620, stdev: 3038.513531, max: 787784.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 32.829360, stdev: 407.369029, max: 151032.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 115.615995, stdev: 4261.692426, max: 656583.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 9.985474, stdev: 102.062806, max: 47678.000000, min: 0.000000), NaN: 0
14 stats: (count: 2391166, mean: 72.629991, stdev: 443.191626, max: 145343.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 20.437082, stdev: 804.888987, max: 136097.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 7.112892, stdev: 76.967212, max: 25587.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 4.875653, stdev: 78.436967, max: 8882.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 39.477453, stdev: 323.215951, max: 183722.000000, min: 0.000000), NaN: 0
访问其他类网站的次数 stats: (count: 2391166, mean: 0.000000, stdev: 0.000000, max: 0.000000, min: 0.000000), NaN: 0
20 stats: (count: 2391166, mean: 66.446228, stdev: 611.911873, max: 568889.000000, min: 0.000000), NaN: 0
     */
    // 屬性正規化
    val norm = new Normalizer()
    val profileNormals: RDD[(String, Vector)] = profileToks.map{ case(imei, features) => (imei, norm.transform(Vectors.dense(features))) }
    NAStat.statsWithMissing(profileNormals.map{ case (imei, features) => features.toArray }).
      foreach(println)
    /*
00 stats: (count: 2391166, mean: 0.003648, stdev: 0.043150, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.000032, stdev: 0.003505, max: 0.820644, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.000494, stdev: 0.021420, max: 0.929880, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.039382, stdev: 0.184361, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.092134, stdev: 0.259398, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.212180, stdev: 0.327933, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.288675, stdev: 0.381518, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.000115, stdev: 0.004942, max: 0.998214, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.003141, stdev: 0.031182, max: 0.999747, min: 0.000000), NaN: 0
09 stats: (count: 2391166, mean: 0.002953, stdev: 0.026723, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.038914, stdev: 0.149384, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.010562, stdev: 0.067241, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.003630, stdev: 0.046494, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.002975, stdev: 0.030421, max: 0.999202, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.029166, stdev: 0.108178, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.008534, stdev: 0.051135, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.003268, stdev: 0.031157, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.001065, stdev: 0.017670, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.015893, stdev: 0.074352, max: 1.000000, min: 0.000000), NaN: 0
19 stats: (count: 2391166, mean: 0.035630, stdev: 0.109365, max: 1.000000, min: 0.000000), NaN: 0
     */
    /*
stats: (count: 2391166, mean: 0.003648, stdev: 0.043150, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.000032, stdev: 0.003505, max: 0.820644, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.000494, stdev: 0.021420, max: 0.929880, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.039382, stdev: 0.184361, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.092134, stdev: 0.259398, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.212180, stdev: 0.327933, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.288675, stdev: 0.381518, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.000115, stdev: 0.004942, max: 0.998214, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.003141, stdev: 0.031182, max: 0.999747, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.002953, stdev: 0.026723, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.038914, stdev: 0.149384, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.010562, stdev: 0.067241, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.003630, stdev: 0.046494, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.002975, stdev: 0.030421, max: 0.999202, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.029166, stdev: 0.108178, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.008534, stdev: 0.051135, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.003268, stdev: 0.031157, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.001065, stdev: 0.017670, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.015893, stdev: 0.074352, max: 1.000000, min: 0.000000), NaN: 0
**stats: (count: 2391166, mean: 0.000000, stdev: 0.000000, max: 0.000000, min: 0.000000), NaN: 0
stats: (count: 2391166, mean: 0.035630, stdev: 0.109365, max: 1.000000, min: 0.000000), NaN: 0

    println(profileNormalLps.take(5).map{ lp => lp.features.size }.mkString(";"))
21;21;21;21;21
     */
    // 建立次數分配
    val featureHists: IndexedSeq[(Int, (Array[Double], Array[Long]))] = (0 to 19).map{ idx => (idx, profileNormals.map{ case(imei, features) => features(idx)}.histogram(5)) }
    /*
    val featureHists: IndexedSeq[(Int, (Array[Double], Array[Long]))] = (0 to 20).map{ idx => (idx, profileNormals.map{ case(imei, features) => features(idx)}.histogram(5)) }
    val featureHistsOld = featureHists
    val buf = featureHists.toBuffer
    buf.remove(featureHists.size-2)
    val featureHists = buf.toIndexedSeq
    val buf = featureHists.toBuffer
    val (idx, ary) = buf.remove(buf.size-1)
    buf+=((idx-1, ary))
    val featureHists = buf.toIndexedSeq
     */
    println (featureHists.map{ case (idx, (cent, height)) =>
      f"[${idx}]" + "="*50 + "\n" +
        " "*2 + cent.mkString(",") + "\n" +
        " "*2 + height.mkString(",") }.mkString("\n") )
    /*
[0]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2380904,4316,1612,1087,3247
[1]==================================================
  0.0,0.16412877836616419,0.32825755673232837,0.49238633509849256,0.656515113464
6567,0.820643891830821
  2391075,39,10,1,41
[2]==================================================
  0.0,0.18597609572663393,0.37195219145326786,0.5579282871799018,0.7439043829065
357,0.9298804786331697
  2389893,4,0,0,1269
[3]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2282966,11312,11729,5515,79644
[4]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2132133,48297,24585,21874,164277
[5]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  1649051,172945,144646,151573,272951
[6]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  1451042,170821,154106,158156,457041
[7]==================================================
  0.0,0.19964276997645847,0.39928553995291693,0.5989283099293754,0.7985710799058339,0.9982138498822923
  2390924,141,64,22,15
[8]==================================================
  0.0,0.19994943607214433,0.39989887214428865,0.599848308216433,0.7997977442885773,0.9997471803607216
  2380626,6007,3389,732,412
[9]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2385806,3126,560,1522,152
[10]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2269781,34771,24495,20669,41450
[11]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2356148,19969,4221,6217,4611
[12]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2382684,722,1650,1731,4379
[13]==================================================
0.0,0.1998403993637648,0.3996807987275296,0.5995211980912943,0.7993615974550592,0.9992019968188239
  2384416,4325,588,339,1498
[14]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2298995,40581,24697,11280,15613
[15]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2368587,13079,5063,822,3615
[16]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2383648,4142,1405,1077,894
[17]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2386474,3098,1431,102,61
[18]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2343437,25414,12085,3942,6288
[19]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2276750,64728,22658,12807,14223
     */
    /*
[0]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2380904,4316,1612,1087,3247
[1]==================================================
  0.0,0.16412877836616419,0.32825755673232837,0.49238633509849256,0.6565151134646567,0.820643891830821
  2391075,39,10,1,41
[2]==================================================
  0.0,0.18597609572663393,0.37195219145326786,0.5579282871799018,0.7439043829065357,0.9298804786331697
  2389893,4,0,0,1269
[3]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2282966,11312,11729,5515,79644
[4]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2132133,48297,24585,21874,164277
[5]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  1649051,172945,144646,151573,272951
[6]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  1451042,170821,154106,158156,457041
[7]==================================================
  0.0,0.19964276997645847,0.39928553995291693,0.5989283099293754,0.7985710799058339,0.9982138498822923
  2390924,141,64,22,15
[8]==================================================
  0.0,0.19994943607214433,0.39989887214428865,0.599848308216433,0.7997977442885773,0.9997471803607216
  2380626,6007,3389,732,412
[9]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2385806,3126,560,1522,152
[10]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2269781,34771,24495,20669,41450
[11]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2356148,19969,4221,6217,4611
[12]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2382684,722,1650,1731,4379
[13]==================================================
  0.0,0.1998403993637648,0.3996807987275296,0.5995211980912943,0.7993615974550592,0.9992019968188239
  2384416,4325,588,339,1498
[14]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2298995,40581,24697,11280,15613
[15]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2368587,13079,5063,822,3615
[16]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2383648,4142,1405,1077,894
[17]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2386474,3098,1431,102,61
[18]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2343437,25414,12085,3942,6288
[19]==================================================
  0.0,0.0
  2391166
[20]==================================================
  0.0,0.2,0.4,0.6,0.8,1.0
  2276750,64728,22658,12807,14223
     */
    // 建立貼標們檻
    val tagThresholds: Map[Int, Double] = sc.broadcast(featureHists.map{ case (idx, (cent: Array[Double], height)) => (idx, cent(4)) }.toMap).value
    /*
    println(tagThresholds.toArray.sortBy{ case (idx, threshold) => idx }.mkString(" "))
(0,0.8) (1,0.6565151134646567) (2,0.7439043829065357) (3,0.8) (4,0.8) (5,0.8) (6,0.8) (7,0.7985710799058339) (8,0.7997977442885773) (9,0.8) (10,0.8) (11,0.8) (12,0.8) (13,0.7993615974550592) (14,0.8) (15,0.8) (16,0.8) (17,0.8) (18,0.8) (19,0.8)
     */
    // 轉換用戶屬性標籤
    var userTags: RDD[userProfile] = profileNormals.mapPartitions{ ite =>
      ite.map{ case (imei, features) =>
        val tags: Array[Double] = tagThresholds.
          map{ case (idx, threshold) => (idx, if (features(idx) > threshold) 1 else 0) }.
          toArray.
          sortBy{ case (idx, flag) => idx }.
          map{ case (idx, flag) => flag.toDouble }
        userProfile(imei, tags) } }.cache()
    userTags.getStorageLevel.useMemory
    /*
    userTags.count() // = 2391166
     */
    // 儲存
    path = "/home/leoricklin/dataset/woplus/usertag/usertag.csv"
    saveUserProfiles(sc, userTags, path)
    /*
    profileTags.count = 2391166
    println( profileTags.take(5).
      map{ case(imei, features) =>
      f"[${imei}][${features.toArray.mkString(",")}]" }.
      mkString("\n") )
[2ab41bf4da442c8b2f3306c28d15d919][0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
[d2aa3932a2e30174c48db0903c5d6be7][0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
[2d2ffd78336770d589a239943a3b8491][0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
[181271cc81d8e27efcf88dd8a67233b6][0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
[1e1528b7c57a948b0c3bd05eec973478][0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
     */
  }

  def reportA(sc:SparkContext) = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    // 載入用戶標籤紀錄
    var path = "file:///home/leoricklin/dataset/woplus/usertag/usertag.csv"
    val userTags = loadUserProfiles(sc, path).cache()
    userTags.getStorageLevel.useMemory
    userTags.count // = 2391166
    println(userTags.take(5).map{ up => userProfile2Str(up)}.mkString("\n"))
    /*
2ab41bf4da442c8b2f3306c28d15d919,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
d2aa3932a2e30174c48db0903c5d6be7,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2d2ffd78336770d589a239943a3b8491,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
181271cc81d8e27efcf88dd8a67233b6,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
1e1528b7c57a948b0c3bd05eec973478,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0     */
    userTags.toDF().registerTempTable("usertag")
    var df = sqlContext.sql("select * from usertag limit 1") //  = [imei: string, features: array<double>]
    var result = df.collect()
    println(result.map{ row => userProfile2Str(row2Userprofile(row))  }.mkString("\n"))
    /*
2ab41bf4da442c8b2f3306c28d15d919,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
     */
    //
    userTags.unpersist(true)

  }

}
