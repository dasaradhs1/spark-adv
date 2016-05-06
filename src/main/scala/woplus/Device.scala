package woplus

import ch04.DTreeUtil
import java.io.File
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import scala.collection.immutable.IndexedSeq
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import tw.com.chttl.spark.mllib.util.NAStat
import tw.com.chttl.spark.mllib.util.DTUtil


/**
 * Created by leorick on 2016/4/28.
 */
object Device {
  val appName = "woplus.device"

  case class Imsi(imsi:String)

  def row2Imsi(row:Row): Imsi = {
    Imsi(row.getString(0))
  }

  def imsi2Str(imsi:Imsi): String = {
    imsi.imsi
  }

  // [0.month: int, 1.imsi: string, 2.net: string, 3.gender: string, 4.age: string,
  // 5.arpu: string, 6.dev_vendor: string, 7.dev_type: string, 8.bytes: string, 9.voice: int,
  // 10.sms: int]
  case class deviceLog(month:Int, imsi:String, net:String, gender:String, age:String,
                    arpu:String, dev_vendor:String, dev_type:String, bytes:String, voice:Int,
                    sms:Int)

  def deviceLog2Str(dev:deviceLog): String = {
    Array(dev.month.toString, dev.imsi, dev.net, dev.gender, dev.age,
      dev.arpu, dev.dev_vendor, dev.dev_type, dev.bytes, dev.voice.toString,
      dev.sms.toString).mkString(",")
  }

  def row2DeviceLog(row:Row): deviceLog = {
    deviceLog(row.getInt(0), row.getString(1), row.getString(2), row.getString(3), row.getString(4),
      row.getString(5), row.getString(6), row.getString(7), row.getString(8), row.getInt(9),
      row.getInt(10))
  }
  
  def deviceLogs2Vector(sc:SparkContext, devices:RDD[deviceLog], mapping: Map[Int, Map[Int, Int]])
  : RDD[(String, Vector)] = {
    val valMapping = sc.broadcast(mapping).value
    devices.mapPartitions{ ite =>
      ite.map{ device => (device.imsi, Vectors.dense(
        valMapping(2)(device.net.hashCode).toDouble,
        valMapping(3)(device.gender.hashCode).toDouble,
        valMapping(4)(device.age.hashCode).toDouble,
        valMapping(5)(device.arpu.hashCode).toDouble,
        valMapping(8)(device.bytes.hashCode).toDouble,
        device.voice.toDouble,
        device.sms.toDouble)) }
    }
  }

  /*
    def rowDeviceToString(row:Row): String = {
      f"[${row.getInt(0)}][${row.getString(1)}][${row.getString(2)}][${row.getString(3)}][${row.getString(4)}]" +
        f"[${row.getString(5)}][${row.getString(6)}][${row.getString(7)}][${row.getString(8)}][${row.getInt(9)}]" +
        f"[${row.getInt(10)}]"
    }
  */

  def bytes2hex(bytes: Array[Byte], sep: Option[String] = None): String = {
    sep match {
      case None => bytes.map("%02x".format(_)).mkString
      case _ => bytes.map("%02x".format(_)).mkString(sep.get)
    }
  }

  def loadDeviceLogSrc(sc:SparkContext, path:String): RDD[String] = {
     sc.textFile(path)
  }

  def loadImsiTargetSrc(sc:SparkContext, path:String): RDD[String] = {
    sc.textFile(path)
  }

  def splitDeviceLogSrc(srcDevice:RDD[String]): RDD[Array[String]] = {
    /* 分隔格式錯誤
    val tokDevice = srcDevice.map{_.split(",")}.
      filter{ case toks => !toks(1).contains("IMSI") }.
      cache
    NAStat.statsWithMissing( tokDevice.map{ toks => Array(toks.size.toDouble)} )
  Array(stats: (count: 6,000,012, mean: 11.008381, stdev: 0.149865, max: 27.000000, min: 11.000000), NaN: 0)
    tokDevice.unpersist(true)
     */
    srcDevice.map{_.split("""","""")}.
      filter{ case toks => !toks(1).contains("IMSI") }
  }

  def tok2DeviceLog(tokens:RDD[Array[String]]): RDD[deviceLog] = {
    tokens.map{ ary => deviceLog(ary(0).replaceAll(""""""","").toInt, ary(1), ary(2), ary(3), ary(4),
      ary(5), ary(6), ary(7), ary(8), ary(9).replaceAll(""""""","").toInt, ary(10).replaceAll(""""""","").toInt) }
  }

  def tok2ImsiTarget(tokens:RDD[String]): RDD[Imsi] = {
    tokens.map{ line => Imsi(line) }
  }

  /**
   *
   * @param sqlContext
   * @param deviceAll
   * @return IndexedSeq[(異動月份:Int, 異動IMSI:DataFrame)] = Vector(
   * (1,None),
   * (2,[imsi: string, n_vendor: string, n_type: string, b_vendor: string, b_type: string]) ....
   */
  def findDiffDev(sqlContext:SQLContext, deviceAll: IndexedSeq[(Int, DataFrame)]): IndexedSeq[(Int, DataFrame)] = {
    deviceAll.
      flatMap{ case (idx, df) => idx match {
        case 1 | 9 =>
          None // 1月沒有前個月資料, (1~8) & (9~12) 月手機格式不同
        case _ => {
          val diff = sqlContext.sql("select a.imsi, nx.dev_vendor as n_vendor, nx.dev_type as n_type, bf.dev_vendor as b_vendor, bf.dev_type as b_type" +
            " from devicetag a" + // 需要预测的IMSI
            f" left join device${idx}%02d   nx on a.imsi = nx.imsi" +
            f" left join device${idx-1}%02d bf on a.imsi = bf.imsi" +
            f" where nx.dev_vendor <> bf.dev_vendor and nx.dev_type <> bf.dev_type")
          Some(idx, diff) } } }
  }

  /**
   *
   * @param sqlContext
   * @param diffDeviceImsis
   * @return IndexedSeq[(異動月份:Int, 前個月行為紀錄:DataFrame)] = Vector(
     (2,[0.month: int,   1.imsi: string,       2.net: string,      3.gender: string, 4.age: string,
         5.arpu: string, 6.dev_vendor: string, 7.dev_type: string, 8.bytes: string,  9.voice: int,
        10.sms: int, 11.label: int]),...
   */
  def prev1MDeviceLabel(sqlContext:SQLContext, diffDeviceImsis:IndexedSeq[(Int, DataFrame)]): IndexedSeq[(Int, DataFrame)] = {
    import sqlContext.implicits._
    diffDeviceImsis.map{ case (idx, diffdevimsi) =>
      diffdevimsi.registerTempTable(f"diffdevimsi${idx}%02d")
      val deviceWithLable = sqlContext.sql("select dev.*, if( isnull(diff.imsi), 0, 1) as label" +
        " from devicetag tg" + // 需要预测的IMSI
        f" left join device${idx-1}%02d    dev  on tg.imsi = dev.imsi" + // 前個月行為
        f" left join diffdevimsi${idx}%02d diff on tg.imsi = diff.imsi") // 當月有換機行為IMSI
      (idx, deviceWithLable) }
  }

  def unionDFs(datasets:IndexedSeq[(Int, DataFrame)]): DataFrame = {
    val ite = datasets.map{ case (idx, df) => df }.iterator
    var dataset: DataFrame = ite.next()
    ite.foreach{ df => dataset = dataset.unionAll(df) }
    dataset
  }
  /*
      val ite = data.map{ case (idx, df) => idx}.iterator
      var dataset = (new ArrayBuffer()).+=(ite.next())
      ite.foreach{ idx => dataset.+=(idx) }
      dataset.mkString(";") //  = 2;3;4;5;6;7;8;10;11;12
   */

  /**
   *
   * @param sc
   * @param dataset
   * @param mapping
   * @return
   *   features: 0.net_mapping,   1.gender_mapping, 2.age_mapping, 3.arpu_mapping, 4.dev_vendor_mapping,
   *             5.bytes_mapping, 6.voice_mapping,  7.sms_mapping
   */
  def prev1MDeviceLabel2LP(sc:SparkContext, dataset:DataFrame, mapping: Map[Int, Map[Int, Int]]): RDD[LabeledPoint] = {
    val valMapping = sc.broadcast(mapping).value
    dataset.mapPartitions{ ite =>
      ite.map{ row =>
        new LabeledPoint(row.getInt(11).toDouble,
          Vectors.dense(
            valMapping(2)(row.getString(2).hashCode).toDouble,
            valMapping(3)(row.getString(3).hashCode).toDouble,
            valMapping(4)(row.getString(4).hashCode).toDouble,
            valMapping(5)(row.getString(5).hashCode).toDouble,
            valMapping(6)(row.getString(6).hashCode).toDouble,
            valMapping(8)(row.getString(8).hashCode).toDouble,
            row.getInt(9).toDouble,
            row.getInt(10).toDouble)) } }
  }

  def saveDiffDeviceImsi(sc:SparkContext, diffDevImsis:DataFrame, path:String): Unit = {
    /*
    diffDevImsis.foreach{ case (idx, df) =>
      df.coalesce(4).write.json(f"${path}/${idx}%02d") }
     */
    import java.io._
    val file = new File(path)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write( diffDevImsis.select("imsi").collect().map{ row => row.getString(0) }.mkString("\n") )
    bw.close()
  }

  def loadDiffDeviceImsi(sc:SparkContext, sqlContext:SQLContext , path:String): Array[(Int, DataFrame)] = {
    /*
    sqlContext.read.json(path)
     */
    // path = "/home/leoricklin/dataset/woplus/device.diff.imsi"
    import sqlContext.implicits._
    new File(path).listFiles().filter(_.isFile).map{ file =>
      ( file.getName.toInt,
        sc.parallelize(
          Source.fromFile(file.getAbsolutePath).getLines().toSeq.map{line => Imsi(line)} ).toDF() ) }
  }

  def saveMapping(sc:SparkContext, mapping:Map[Int,Int], path:String, name:String) = {
    import java.io._
    val file = new File(f"${path}/${name}")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write( mapping.toArray.map{ case (code, idx) => f"${code},${idx}" }.mkString("\n") )
    bw.close()
  }

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    // var path = "file:///home/leo/woplus/device/"
    /*
"�·�","IMSI","���","�Ա�","����ֵ��","ARPUֵ��","�ն�Ʒ��","�ն��ͺ�","����ʹ����","����ͨ��ʱ��","��������
"201503","62a26614d2c506ffa5132da309c2e908","3G","��","40-49","0-49","LG","LG-F200L","0-499","270","35"
"201503","829c63f15407c8431f5b3e19d982a9e3","3G","��","40-49","0-49","Samsung","SM-G3812","0-499","297","25"
"201503","3208398d877b6410b3038a68b95cf659","3G","Ů","40-49","100-149","Apple","iPhone 6 (A1586)","0-499","363","67"
"201503","a929356b003b8682031541f716c2e919","3G","��","50-59","50-99","Apple","iPhone 4S","0-499","1126","11"
     */
    // 需要预测的IMSI
    var path = "file:///home/leoricklin/dataset/woplus/target/"
    // path = "file:///home/leo/woplus/target/"
    val deviceTags = tok2ImsiTarget(loadImsiTargetSrc(sc, path)).cache()
    deviceTags.getStorageLevel.useMemory
    deviceTags.toDF().registerTempTable("devicetag")
    var df = sqlContext.sql("select count(1) from devicetag")
    var result = df.collect()
    /*
    println(f"#rows=${srcDeviceTag.count}") // = 360698
    println(f"distinct #rows=${srcDeviceTag.distinct.count}") // = 360698
    srcDeviceTag.take(5).foreach(println)
24e2a235ebdb9b02b76a9ae749abfd59
6ee517b2c0df31d2ea4a9424f4d9f124
daf2fa5030cbc1464d41da7c3b6cb462
37ae18b2bf8150e63f690ed0449f220f
d022fc28d9bce0fae7d57831f3501260
     */
    // 讀取完整資料
    path = "file:///home/leoricklin/dataset/woplus/device/"
    val deviceLogAll = tok2DeviceLog( splitDeviceLogSrc( loadDeviceLogSrc(sc, path) )).toDF().cache()
    /*
    deviceLogAll.take(5).map{ (row) => rowDeviceToString(row) }.mkString("\n")
[201504][ef6edca386039fa78789676a839eb373][3G][��][18-22][150-199][Xiaomi][Redmi 1S][1500-1999][398][62]
[201504][5cb11710f85819f88c24fa1b64ae1395][3G][Ů][26-29][50-99][Apple][iPhone 4S][1000-1499][918][148]
[201504][a82206d5008a4798072c23d9fa4581ef][3G][��][50-59][0-49][Samsung][SM-G7106][0-499][54][6]
[201504][184f362accc77ebccb0cad595b6abf9d][3G][Ů][26-29][0-49][EBEST][F6][0-499][73][14]
[201504][06bca9179b601cbd836fecda57998730][3G][��][40-49][50-99][Xiaomi][Redmi Note/MI 3S/Redmi][0-499][242][0]
    // 欄位數統計
    NAStat.statsWithMissing( deviceLogAll.map{ toks => Array(toks.size.toDouble)} )
Array(stats: (count: 6000000, mean: 11.000000, stdev: 0.000000, max: 11.000000, min: 11.000000), NaN: 0)
    // 月份
    tksDevice.map( ary => ary(0) ).distinct().collect().foreach(println)
"201504 "201505 "201506 "201510 "201507 "201508 "201511 "201509 "201512 "201501 "201502 "201503
    // IMSI
    tksDevice.map( ary => ary(1) ).distinct().count // = 500000
    // 网别
    tksDevice.map( ary => ary(2) ).distinct().collect().foreach(println)
2G 3G
    // 性别
    tksDevice.map( ary => bytes2hex(ary(3).getBytes) ).distinct().collect().foreach(println)
efbfbdefbfbdefbfbdefbfbd
efbfbdefbfbd
c5ae
    tksDevice.filter( ary => (ary(0).contains("201501") && ary(1).contains("7d8336b84af2ae9e13f2abbad905814e"))
      || (ary(0).contains("201506") && ary(1).contains("ea88b1c34e31af58b1f3e09c84c576b9")) ).
      collect().
      foreach( ary => println(f"${ary(0)},${ary(1)},${bytes2hex(ary(3).getBytes)}"))
"201501,7d8336b84af2ae9e13f2abbad905814e,efbfbdefbfbd // 男
"201506,ea88b1c34e31af58b1f3e09c84c576b9,c5ae         // 女
    // 年龄值段
    tksDevice.map( ary => ary(4) ).distinct().collect().foreach(it => println(f"[${bytes2hex(it.getBytes)}][${it}]"))
[3137efbfbdefbfbdefbfbdefbfbdefbfbdefbfbd][17������]
[31382d3232][18-22]
[32332d3235][23-25]
[32362d3239][26-29]
[33302d3339][30-39]
[34302d3439][40-49]
[35302d3539][50-59]
[3630efbfbdefbfbdefbfbdefbfbd][60����]
[ceb4d6aa][δ֪]
    // ARPU值段
    tksDevice.map( ary => ary(5) ).distinct().collect().foreach(it => println(f"[${bytes2hex(it.getBytes)}][${it}]"))
[][]    <<<<<<<<<<<<<<<<<<<<<<<<<<============================ NaN, 待預測IMSI未出現
[302d3439][0-49]
[35302d3939][50-99]
[3130302d313439][100-149]
[3135302d313939][150-199]
[3230302d323439][200-249]
[3235302d323939][250-299]
[333030efbfbdefbfbdefbfbdefbfbdefbfbdefbfbd][300������]
    // 终端品牌
    tksDevice.map( ary => ary(6) ).distinct().count // = 1756
    // 终端型号
    tksDevice.map( ary => ary(7) ).distinct().count // = 27397
    // 流量使用量
    tksDevice.map( ary => ary(8) ).distinct().collect().foreach(it => println(f"[${bytes2hex(it.getBytes)}][${it}]"))
[][]  <<<<<<<<<<<<<<<<<<<<<<<<<<============================ NaN, 待預測IMSI未出現
[302d343939][0-499]
[3530302d393939][500-999]
[313030302d31343939][1000-1499]
[313530302d31393939][1500-1999]
[323030302d32343939][2000-2499]
[323530302d32393939][2500-2999]
[333030302d33343939][3000-3499]
[333530302d33393939][3500-3999]
[343030302d34343939][4000-4499]
[343530302d34393939][4500-4999]
[35303030efbfbdefbfbdefbfbdefbfbd][5000����]
    // 语音通话时长
    tksDevice.map( ary => ary(9) ).distinct().count = 6075
    // 短信条数
    tksDevice.map( ary => ary(10) ).distinct().count = 2058
    //
    NAStat.statsWithMissing( tksDevice.map{ toks => Array(toks(9).replaceAll(""""""","").toDouble,toks(10).replaceAll(""""""","").toDouble)} )
Array(stats: (count: 6000000, mean: 267.869260, stdev: 416.809862, max: 20078.000000, min: 0.000000), NaN: 0,
      stats: (count: 6000000, mean: 16.641471, stdev: 57.261537, max: 16055.000000, min: 0.000000), NaN: 0)
     */
    // 建立屬性對應表
    result = deviceLogAll.select($"net").distinct().collect()
    val netMap: Map[Int, Int] = result.map{ row => row.getString(0).hashCode }.zipWithIndex.toMap
    /*
    result.foreach{ row =>
      val value = row.getString(0)
      println(f"[${value}][${bytes2hex(value.getBytes)}][${value.hashCode}]") }
     */
    /*
[2G}][3247][1621]
[3G}][3347][1652]
Map(1621 -> 0, 1652 -> 1)
     */
    result = deviceLogAll.select($"gender").distinct().collect()
    val genderMap = result.map{ row => row.getString(0).hashCode }.zipWithIndex.toMap
    /*
    result.foreach{ row =>
      val value = row.getString(0)
      println(f"[${value}][${bytes2hex(value.getBytes)}][${value.hashCode}]") }
     */
    /*
[efbfbdefbfbd][2097056]
[c5ae][366]
[efbfbdefbfbdefbfbdefbfbd][2017367872]
Map(2097056 -> 0, 366 -> 1, 2017367872 -> 2)
     */
    result = deviceLogAll.select($"age").distinct().collect()
    val ageMap = result.map{ row => row.getString(0).hashCode }.zipWithIndex.toMap
    /*
    result.foreach{ row =>
      val value = row.getString(0)
      println(f"[${value}][${bytes2hex(value.getBytes)}][${value.hashCode}]") }
     */
    /*
[17������}][3137efbfbdefbfbdefbfbdefbfbdefbfbdefbfbd][-1566173050]
[18-22}][31382d3232][46965670]
[23-25}][32332d3235][47740239]
[26-29}][32362d3239][47829616]
[30-39}][33302d3339][48574422]
[40-49}][34302d3439][49497974]
[50-59}][35302d3539][50421526]
[60����}][3630efbfbdefbfbdefbfbdefbfbd][-687296262]
[δ֪}][ceb4d6aa][30838]
Map(47740239 -> 7, 30838 -> 4, -687296262 -> 3, -1566173050 -> 0, 48574422 -> 6, 49497974 -> 2, 47829616 -> 5, 46965670 -> 8, 50421526 -> 1)
    */
    result = deviceLogAll.select($"arpu").distinct().collect()
    val arpuMap = result.map{ row => row.getString(0).hashCode }.zipWithIndex.toMap
    /*
    result.foreach{ row =>
      val value = row.getString(0)
      println(f"[${value}][${bytes2hex(value.getBytes)}][${value.hashCode}]") }
     */
    /*
[100-149}][3130302d313439][1957925018]
[300������}][333030efbfbdefbfbdefbfbdefbfbdefbfbdefbfbd][1377528339]
[200-249}][3230302d323439][-1449537636]
[150-199}][3135302d313939][2101070928]
[50-99}][35302d3939][50421650]
[}][][0]
[0-49}][302d3439][1474882]
[250-299}][3235302d323939][-1306391726]
Map(0 -> 5, -1306391726 -> 7, 2101070928 -> 3, -1449537636 -> 2, 50421650 -> 4, 1377528339 -> 1, 1957925018 -> 0, 1474882 -> 6)
     */
    result = deviceLogAll.select($"dev_vendor").distinct().collect()
    val dev_vendorMap = result.map{ row => row.getString(0).hashCode }.zipWithIndex.toMap
    /*
    result.foreach{ row =>
      val value = row.getString(0)
      println(f"[${value}][${bytes2hex(value.getBytes)}][${value.hashCode}]") }
     */
    // dev_vendorMap.size = 1756
    result = deviceLogAll.select($"dev_type").distinct().collect()
    val dev_typeMap = result.map{ row => row.getString(0).hashCode }.zipWithIndex.toMap
    /*
    result.foreach{ row =>
      val value = row.getString(0)
      println(f"[${value}][${bytes2hex(value.getBytes)}][${value.hashCode}]") }
     */
    // dev_typeMap.size = 27343
    /*
    result.foreach{ row =>
      val value = row.getString(0)
      println(f"[${value}][${bytes2hex(value.getBytes)}][${value.hashCode}]") }
     */
    result = deviceLogAll.select($"bytes").distinct().collect()
    val bytesMap = result.map{ row => row.getString(0).hashCode }.zipWithIndex.toMap
    /*
[4000-4499}][343030302d34343939][-674939055]
[1500-1999}][313530302d31393939][576489553]
[2000-2499}][323030302d32343939][-1355057007]
[3000-3499}][333030302d33343939][1132485617]
[500-999}][3530302d393939][1212980289]
[4500-4999}][343530302d34393939][-550817167]
[0-499}][302d343939][45721399]
[1000-1499}][313030302d31343939][452367665]
[5000����}][35303030efbfbdefbfbdefbfbdefbfbd][966067099]
[2500-2999}][323530302d32393939][-1230935119]
[3500-3999}][333530302d33393939][1256607505]
[}][][0]
Map(0 -> 11, -674939055 -> 0, -1355057007 -> 2, 1132485617 -> 3, 45721399 -> 6, 576489553 -> 1, 452367665 -> 7, 1212980289 -> 4, -550817167 -> 5, -1230935119 -> 9, 1256607505 -> 10, 966067099 -> 8)
     */
    val mapping: Map[Int, Map[Int, Int]] = Map(0 -> Map(0->0), 1->Map(0->0), 2->netMap, 3->genderMap, 4->ageMap,
      5->arpuMap, 6->dev_vendorMap, 7->dev_typeMap, 8->bytesMap )
    //
    deviceLogAll.unpersist(true)
    // 分析 11/12月
    val analy1112 = {
      path = "file:///w.data/WORKSPACE.2/dataset/woplus/device/201511.csv"
      val device11 = tok2DeviceLog(splitDeviceLogSrc(loadDeviceLogSrc(sc, path))) // : DataFrame =
      device11.toDF().registerTempTable("device11")
      /*
      df = sqlContext.sql("select count(1) from device11")
      result = df.collect()
      result.foreach{ row => println(row.get(0)) } // = 500000
       */
      // path = "file:///w.data/WORKSPACE.2/dataset/woplus/device/201512.csv"
      path = "file:///home/leoricklin/dataset/woplus/device/201512.csv"
      val device12 = tok2DeviceLog( splitDeviceLogSrc( loadDeviceLogSrc(sc, path))).cache()
      device12.toDF().registerTempTable("device12")
      device12.getStorageLevel.useMemory
      /*
      df = sqlContext.sql("select * from device12 limit 5")
      result = df.collect()
      println(result.map{ row => device2Str(row2Device(row)) }.mkString("\n"))
201512,89b4ad5cd0d3cdeac177ca0bd2170786,3G,ï¿½ï¿½,30-39,50-99,ï¿½ï¿½ï¿½ï¿½,Coolpad T2-W01,0-499,494,93
201512,795dc714d902acf9830f34ed2bcd1054,3G,ï¿½ï¿½,30-39,0-49,ï¿½ï¿½Îª,Che1-CL10,0-499,476,1
201512,8ae17bd692cb4dd11d768d2ae1b10fa7,3G,ï¿½ï¿½,60ï¿½ï¿½ï¿½ï¿½,150-199,Æ»ï¿½ï¿½,A1431,0-499,1601,87
201512,ad4a0e92fb31bbb3c797fb4c3b6bc9a2,3G,ï¿½ï¿½,40-49,300ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½,ï¿½ï¿½Îª,H60-L02,0-499,4,2
201512,0664611530f4443654c87abac1c49169,3G,ï¿½ï¿½,40-49,300ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½,Æ»ï¿½ï¿½,A1528,1000-1499,927,176
       */
      df = sqlContext.sql("select a.imsi, b.dev_vendor as n_vendor, b.dev_type as n_type, c.dev_vendor as b_vendor, c.dev_type as b_type from devicetag a left join device12 b on a.imsi=b.imsi left join device11 c on a.imsi = c.imsi")
      /*
      : DataFrame = [imsi: string, n_vendor: string, n_type: string, b_vendor: string, b_type: string]
      df.count() // 360698
      result = df.take(5)
      result.foreach{ row =>
        val flag = row.getString(1).equals(row.getString(3)) && row.getString(2).equals(row.getString(4))
        println(f"[${row.getString(0)}][${row.getString(1)}][${row.getString(2)}][${row.getString(3)}][${row.getString(4)}][${flag}]") }
  [00085b5c30ec953e28e18820646d0f66][����][M460][����][M460][true]
  [004c6d0c7a83b1a07339e8e5e4bee881][��Ϊ][HUAWEI P6-C00][��Ϊ][HUAWEI P6-C00][true]
  [0051c5a9851964842546a77c4911eaa2][ƻ��][A1325][ƻ��][A1325][true]
  [0060de21cbd915f9f792de90130e3050][΢��][Nokia 501][ŵ����][Nokia 501][false]
  [0065a1268ab2e710fbf73d896b0aa64d][���][PI39200][���][PI39200][true]
       */
      df.registerTempTable("devtag1112")
      df = sqlContext.sql("select * from devtag1112 where n_vendor <> b_vendor and n_type <> b_type")
      /*
      : DataFrame = [imsi: string, n_vendor: string, n_type: string, b_vendor: string, b_type: string]
      diff1112.count() // 15143
      result = diff1112.take(5)
      result.foreach{ row =>
        val flag = row.getString(1).equals(row.getString(3)) && row.getString(2).equals(row.getString(4))
        println(f"[${row.getString(0)}][${row.getString(1)}][${row.getString(2)}][${row.getString(3)}][${row.getString(4)}][${flag}]") }
  [012fc2e11d3b434a9f0e47c694a0c36d][΢��][Nokia 1050][С��][2012061][false]
  [0570a49afa3e741766d65c09444e82ab][�Ѷ][C520Lw][ƻ��][A1524][false]
  [07273f7a228768e8d2e6f79ce641ff1b][Ŭ����][NX511J][ƻ��][A1524][false]
  [09a1f55c05f003338386daf747f9345e][С��][2013122][����][ZTE V987][false]
  [0b8b59ba27c0843a4367e6e12ad3cf8e][��Ϊ][HUAWEI RIO-AL00][����][GT-I9152][false]
       */
    }
    // 讀取各月檔案並轉為DF
    val deviceLogEachMonth: IndexedSeq[(Int, DataFrame)] = (1 to 12).map{ idx =>
      // path = f"file:///w.data/WORKSPACE.2/dataset/woplus/device/2015${idx}%02d.csv"
      path = f"file:///home/leoricklin/dataset/woplus/device/2015${idx}%02d.csv"
      val df = tok2DeviceLog( splitDeviceLogSrc( loadDeviceLogSrc(sc, path))).toDF()
      df.registerTempTable(f"device${idx}%02d")
      (idx, df) } // : IndexedSeq[(月份:Int, 行為紀錄:DataFrame)
    // 從12月開始, 找出當月與上個月手機不同的用戶
    val diffDeviceImsis = findDiffDev(sqlContext, deviceLogEachMonth)
    /*
    diffDeviceImsis.map{ case (idx, df) => f"[${idx}%02d]${df.count}" }.mkString("\n")
[02]22027
[03]21892
[04]18105
[05]19652
[06]20207
[07]18841
[08]18750
[10]17591
[11]34139
[12]15143
    diffDeviceImsis.filter{ case (idx, df) => idx == 12}.head._2.
      select("*").limit(5).collect().map{ case row =>
        Array(row.getString(0),row.getString(1),row.getString(2),row.getString(3),row.getString(4)).mkString(",") }.
      mkString("\n")
00a50417a50980b4aff0905c95e8d5a6,Nokia,C1-00,Samsung,GT-E1200M
00e6ed831b497850de796afb17a98c8f,Lenovo,A680/A880,Coolpad,Coolpad W706+
0103999650d29c4cad91dd4549f7df4c,HUAWEI,Y210-0010,KNIGHT,ZX338
01b5018b0be70c4eae46ed00eef0fa96,HUAWEI,Honor 6,Xiaomi,MI 2
02e391138c1750d590455e98a50edc55,Apple,iPhone 4S,Samsung,GT-N7108
     */
    // 儲存, coalesce(3) => job failed
    path = "/home/leoricklin/dataset/woplus/device.diff.imsi"
    diffDeviceImsis.foreach { case (idx, df) =>
      saveDiffDeviceImsi(sc, df, f"${path}/${idx}%02d") }
    /*
    diffDevices.filter{ case (idx, df) => idx==12}.
      map{ case (idx, df) =>
        val cnt = if (df.isEmpty) 0 else df.get.count()
        (idx, cnt) }.
      foreach{ case (idx, cnt) => println(f"[${idx}][${cnt}]") }
[12][15143]

    diffs.foreach{ case (idx, df) => df match {
      case None => println(idx)
      case _ => {
        println(idx)
        println(df.get.sample(false, 0.1).take(5).map{ row => f"[${row.getString(0)}][${row.getString(1)}][${row.getString(2)}][${row.getString(3)}][${row.getString(4)}]" }.mkString("\n"))
      } } }
     */
    // 載入
    val newdiffDeviceImsis = loadDiffDeviceImsi(sc, sqlContext, path)
    /*
    println(newdiffDeviceImsis.map{ case (idx, df) => f"[${idx}%02d]${df.count()}" }.mkString("\n"))
[05]19652
[10]17591
[04]18105
[06]20207
[03]21892
[11]34139
[08]18750
[02]22027
[12]15143
[07]18841
     */
    // 準備模型 dataset
    val datasets = prev1MDeviceLabel(sqlContext, diffDeviceImsis)
    // 儲存, coalesce(<16) => job failed
    /*
    datasets.foreach{ case (idx, df) =>
      // path = f"file:///w.data/WORKSPACE.2/dataset/woplus/device.diff.data/2015${idx}%02d.json"
      path = f"file:///home/leoricklin/dataset/woplus/device.diff.data/2015${idx}%02d.json"
      df.coalesce(64).write.json(path) }
     */
    // 驗證
    /*
    println(datasets.filter{ case (idx, df) => idx == 12 }.
      map{ case (idx, df) => df }.
      head.
      where($"label" === 1).
      take(5).
      map{ row =>
        f"[${row.getString(2).hashCode}][${netMap(row.getString(2).hashCode)}]" +
          f"[${row.getString(3).hashCode}][${genderMap(row.getString(3).hashCode)}]" +
          f"[${row.getString(4).hashCode}][${ageMap(row.getString(4).hashCode)}]" +
          f"[${row.getString(5).hashCode}][${arpuMap(row.getString(5).hashCode)}]" +
          f"[${row.getString(6).hashCode}][${dev_vendorMap(row.getString(6).hashCode)}]" +
          f"[${row.getString(7).hashCode}][${dev_typeMap(row.getString(7).hashCode)}]" +
          f"[${row.getString(8).hashCode}][${bytesMap(row.getString(8).hashCode)}]" +
          f"[${row.getInt(9)}]" +
          f"[${row.getInt(10)}]" }.
      mkString("\n"))
[1652][1][2097056][0][-1566173050][0][1957925018][0][3112833][1089][-1448467028][10371][45721399][6][919][37]
[1652][1][366][1][47829616][5][-1449537636][2][2522779][750][61541159][10806][1212980289][4][164][13]
[1652][1][2097056][0][-1566173050][0][1377528339][1][2522779][750][61541159][10806][452367665][7][380][5]
[1621][0][2097056][0][50421526][1][1474882][6][2017367872][1004][1482588055][22156][45721399][6][11][1]
[1652][1][2097056][0][49497974][2][50421650][4][2017367872][1004][2078823070][24118][45721399][6][22][5]
     */
    /*
    println(datasets.filter{ case (idx, df) => idx == 12 }.
      map{ case (idx, df) => df }.
      head.
      where($"label" === 1).count())
15143
     */
    /*
    var dfs = data.filter{case (idx, df) => idx == 12}
    dfs.map{ case (idx, df) =>
      val diff = df.where($"label" === 1)
      println(f"[${idx}][${diff.count()}]")
      println(diff.take(5).map{ row =>
        f"[${row.getInt(0)}][${row.getString(1)}][${row.getString(2)}][${row.getString(3)}][${row.getString(4)}]" +
        f"[${row.getString(5)}][${row.getString(6)}][${row.getString(7)}][${row.getString(8)}][${row.getInt(9)}]" +
        f"[${row.getInt(10)}][${row.getInt(11)}]"}.mkString("\n")) }
[12][15143]
[201511][012fc2e11d3b434a9f0e47c694a0c36d][3G][��][17������][100-149][С��][2012061][0-499][919][37][1]
[201511][0570a49afa3e741766d65c09444e82ab][3G][Ů][26-29][200-249][ƻ��][A1524][500-999][164][13][1]
[201511][07273f7a228768e8d2e6f79ce641ff1b][3G][��][17������][300������][ƻ��][A1524][1000-1499][380][5][1]
[201511][09a1f55c05f003338386daf747f9345e][2G][��][50-59][0-49][����][ZTE V987][0-499][11][1][1]
[201511][0b8b59ba27c0843a4367e6e12ad3cf8e][3G][��][40-49][50-99][����][GT-I9152][0-499][22][5][1]

    result = sqlContext.sql("select nx.imsi, nx.dev_vendor as n_vendor, nx.dev_type as n_type, bf.dev_vendor as b_vendor, bf.dev_type as b_type" +
      " from device12 nx inner join device11 bf on nx.imsi = bf.imsi" +
      " where nx.imsi in ('012fc2e11d3b434a9f0e47c694a0c36d','0570a49afa3e741766d65c09444e82ab','07273f7a228768e8d2e6f79ce641ff1b','09a1f55c05f003338386daf747f9345e','0b8b59ba27c0843a4367e6e12ad3cf8e')").
      collect()
    result.foreach{ row => println(f"[${row.getString(0)}][${row.getString(1)}][${row.getString(2)}][${row.getString(3)}][${row.getString(4)}]") }
[012fc2e11d3b434a9f0e47c694a0c36d][΢��][Nokia 1050][С��][2012061]
[0570a49afa3e741766d65c09444e82ab][�Ѷ][C520Lw][ƻ��][A1524]
[07273f7a228768e8d2e6f79ce641ff1b][Ŭ����][NX511J][ƻ��][A1524]
[09a1f55c05f003338386daf747f9345e][С��][2013122][����][ZTE V987]
[0b8b59ba27c0843a4367e6e12ad3cf8e][��Ϊ][HUAWEI RIO-AL00][����][GT-I9152]

    dfs.map{ case (idx, df) =>
      val diff = df.
        select(when($"arpu".isNaN, "NaN").
          when($"arpu".isNull, "NaN").
          otherwise($"arpu")).
        map{ row => row.getString(0).hashCode }.
        distinct().collect().mkString(";")
      println(f"${diff}") }
net: 1621;1652
gender: 2097056;2017367872;366
age: 47829616;48574422;30838;47740239;46965670;50421526;-687296262;-1566173050;49497974
arpu:1957925018;50421650;-1306391726;1474882;2101070928;1377528339;-1449537636
    */
    /*
    var dfs = data.map{  case (idx, df) =>
      val newdf = df.
        // select($"arpu").
        select(when($"arpu".isNaN, "NaN").when($"arpu".isNull, "NaN").otherwise($"arpu")).
        distinct()
      (idx, newdf)
    }
    dfs.foreach{ case (idx, df) =>
      val out = df.collect().map{row => row.get(0).toString}.mkString(";")
      println(f"[${idx}][${out}]")  }
arpu:
[2][100-149;300������;200-249;150-199;50-99;0-49;250-299]
[3][100-149;300������;200-249;150-199;50-99;0-49;250-299]
[4][100-149;300������;200-249;150-199;50-99;0-49;250-299]
[5][100-149;300������;200-249;150-199;50-99;0-49;250-299]
[6][100-149;300������;200-249;150-199;50-99;0-49;250-299]
[7][100-149;300������;200-249;150-199;50-99;0-49;250-299]
[8][100-149;300������;200-249;150-199;50-99;0-49;250-299]
[10][100-149;300������;200-249;150-199;50-99;0-49;250-299]
[11][100-149;300������;200-249;150-199;50-99;0-49;250-299]
[12][100-149;300������;200-249;150-199;50-99;0-49;250-299]
bytes:
 [2][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
 [3][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
 [4][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
 [5][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
 [6][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
 [7][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
 [8][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
[10][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
[11][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
[12][4000-4499;1500-1999;2000-2499;3000-3499;500-999;4500-4999;0-499;1000-1499;5000����;2500-2999;3500-3999]
     */
    // 建立 labeledpoint
    val lps = prev1MDeviceLabel2LP(sc, unionDFs(datasets), mapping).cache()
    /*
    lps.take(5).map{lp => println(f"[${lp.label}][${lp.features.size}][${lp.features.toArray.mkString(",")}]")}
[0.0][8][0.0,0.0,6.0,6.0,1289.0,6.0,0.0,0.0]
[0.0][8][0.0,1.0,6.0,6.0,1228.0,6.0,666.0,0.0]
[0.0][8][1.0,0.0,3.0,6.0,493.0,6.0,278.0,0.0]
[0.0][8][0.0,0.0,6.0,4.0,1228.0,6.0,116.0,0.0]
[0.0][8][0.0,1.0,3.0,6.0,202.0,6.0,55.0,16.0]
     */
    NAStat.statsWithMissing( lps.map{ lp => new ArrayBuffer[Double]().+=(lp.label).++=(lp.features.toArray).toArray } ).
      foreach(it => println(it))
    /*
stats: (count: 3606980, mean: 0.057208, stdev: 0.232239, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 0.376015, stdev: 0.484384, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 0.283296, stdev: 0.491824, max: 2.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 4.245268, stdev: 2.230167, max: 8.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 4.796174, stdev: 1.907042, max: 7.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 741.728691, stdev: 390.014259, max: 1755.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 5.858392, stdev: 0.713862, max: 10.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 267.962409, stdev: 413.240798, max: 17420.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 17.260870, stdev: 57.879806, max: 16055.000000, min: 0.000000), NaN: 0
     */
    /*
label:  stats: (count: 3606980, mean: 0.057208, stdev: 0.232239, max: 1.000000, min: 0.000000), NaN: 0
net:    stats: (count: 3606980, mean: 0.376015, stdev: 0.484384, max: 1.000000, min: 0.000000), NaN: 0
gender: stats: (count: 3606980, mean: 0.283296, stdev: 0.491824, max: 2.000000, min: 0.000000), NaN: 0
age:    stats: (count: 3606980, mean: 4.245268, stdev: 2.230167, max: 8.000000, min: 0.000000), NaN: 0
arpu:   stats: (count: 3606980, mean: 4.796174, stdev: 1.907042, max: 7.000000, min: 0.000000), NaN: 0
vendor: stats: (count: 3606980, mean: 739.833648, stdev: 389.904231, max: 1754.000000, min: 0.000000), NaN: 0
type:   stats: (count: 3606980, mean: 14571.360523, stdev: 7803.198309, max: 27394.000000, min: 0.000000), NaN: 0
byte:   stats: (count: 3606980, mean: 5.858392, stdev: 0.713862, max: 10.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 267.962409, stdev: 413.240798, max: 17420.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 17.260870, stdev: 57.879806, max: 16055.000000, min: 0.000000), NaN: 0
     */
    // features: net, gender, age, arpu, dev_vendor, bytes, voice, sms
    val newlps: RDD[LabeledPoint] = lps.
      map{ lp =>
        val features = lp.features.toArray
        new LabeledPoint(lp.label,
          Vectors.dense(
            features(0), features(1),features(2),features(3),features(5),features(6),features(7))) }.
      cache
    newlps.first().features.size // 7
    NAStat.statsWithMissing( newlps.map{ lp => new ArrayBuffer[Double]().+=(lp.label).++=(lp.features.toArray).toArray } ).
      foreach(it => println(it))
    /*
stats: (count: 3606980, mean: 0.057208, stdev: 0.232239, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 0.376015, stdev: 0.484384, max: 1.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 0.283296, stdev: 0.491824, max: 2.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 4.245268, stdev: 2.230167, max: 8.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 4.796174, stdev: 1.907042, max: 7.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 5.858392, stdev: 0.713862, max: 10.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 267.962409, stdev: 413.240798, max: 17420.000000, min: 0.000000), NaN: 0
stats: (count: 3606980, mean: 17.260870, stdev: 57.879806, max: 16055.000000, min: 0.000000), NaN: 0
    newlps.take(5).map{lp => println(f"[${lp.label}][${lp.features.size}][${lp.features.toArray.mkString(",")}]")}
     */
    // fit model
    val numClasses = 2
    /*
    val catInfo = Map(2->2, // net
      3->3, // gender
      4->9, // age
      5->8, // arpu
      6->1756, // dev_vendor
      7->27397, // dev_type
      8->11) // bytes
     */
    val catInfo = Map(
      0->2, // net
      1->3, // gender
      2->9, // age
      3->8, // arpu
      4->11) // bytes
    val maxBins = Array(50)
    val maxDepth = Array(30)
    val impurities = Array(1)
    val maxTrees = Array(32)
    val numFolds = 3
    val seed = 1
    val cvModels: Array[(Array[Int], Array[(RandomForestModel, Array[Double])])] = DTUtil.multiParamRfCvs(newlps, numClasses, catInfo, maxBins, maxDepth, impurities, maxTrees, numFolds)
    /*
    1.0047646157836E7ms
: Array[(Array[Int], Array[(RandomForestModel, Array[Double])])]
= Array((Array(50, 30, 1, 32),Array(
  (TreeEnsembleModel classifier with 32 trees, Array(0.942779316482438, 0.942779316482438, 0.942779316482438)),
  (TreeEnsembleModel classifier with 32 trees, Array(0.9424896684170594, 0.9424896684170594, 0.9424896684170594)),
  (TreeEnsembleModel classifier with 32 trees ,Array(0.9429902582240717, 0.9429902582240717, 0.9429902582240717)))))
     */
    // 儲存模型
    cvModels.head._2.zipWithIndex.map(_.swap).foreach{ case (idx, (model, evals)) =>
      path = f"file:///home/leoricklin/dataset/woplus/device.rfmodel.${idx}%02d"
      model.save(sc, path) }
    // 預測12月用戶
    path = "file:///home/leoricklin/dataset/woplus/device/201512.csv"
    val device12 = tok2DeviceLog( splitDeviceLogSrc( loadDeviceLogSrc(sc, path))).cache()
    device12.toDF().registerTempTable("device12")
    device12.getStorageLevel.useMemory
    df = sqlContext.sql("select * from device12 limit 5") // = [month: int, imsi: string, net: string, gender: string, age: string, arpu: string, dev_vendor: string, dev_type: string, bytes: string, voice: int, sms: int]
    result = df.collect()
    println(result.map{ row => deviceLog2Str(row2DeviceLog(row)) }.mkString("\n"))
    /*
201512,89b4ad5cd0d3cdeac177ca0bd2170786,3G,ï¿½ï¿½,30-39,50-99,ï¿½ï¿½ï¿½ï¿½,Coolpad T2-W01,0-499,494,93
201512,795dc714d902acf9830f34ed2bcd1054,3G,ï¿½ï¿½,30-39,0-49,ï¿½ï¿½Îª,Che1-CL10,0-499,476,1
201512,8ae17bd692cb4dd11d768d2ae1b10fa7,3G,ï¿½ï¿½,60ï¿½ï¿½ï¿½ï¿½,150-199,Æ»ï¿½ï¿½,A1431,0-499,1601,87
201512,ad4a0e92fb31bbb3c797fb4c3b6bc9a2,3G,ï¿½ï¿½,40-49,300ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½,ï¿½ï¿½Îª,H60-L02,0-499,4,2
201512,0664611530f4443654c87abac1c49169,3G,ï¿½ï¿½,40-49,300ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½,Æ»ï¿½ï¿½,A1528,1000-1499,927,176
     */
    deviceTags.getStorageLevel.useMemory
    deviceTags.toDF().registerTempTable("devicetag")
    df = sqlContext.sql("select * from devicetag limit 5") // = [imsi: string]
    result = df.collect()
    println(result.map{ row => imsi2Str(row2Imsi(row)) }.mkString("\n"))
    val deviceTag12Log: RDD[deviceLog] = sqlContext.
      sql("select dev.* from devicetag tag" +
        " inner join device12 dev" +
        " on tag.imsi = dev.imsi").
      map{ row => row2DeviceLog(row) }

    val deiveTag12Vecs: RDD[(String, Vector)] = deviceLogs2Vector(sc, deviceTag12Log, mapping)
    deiveTag12Vecs.first._2.size // = 7
    deiveTag12Vecs.cache
    val model: RandomForestModel = sc.broadcast(cvModels.head._2.head._1).value
    val deiveTag12Pred: RDD[(String, Vector, Double)] = model.predict{ deiveTag12Vecs.map{ case (imsi, vec) => vec} }.zip(deiveTag12Vecs).
      map{ case (pred, (imsi, vec)) => (imsi, vec, pred) }
    deiveTag12Pred.count() //  = 360698
    //
    path = "/home/leoricklin/dataset/woplus/device.predict/predict.csv"
    import java.io._
    val file = new File(path)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write( deiveTag12Pred.map{ case (imsi, vec, pred) => f"${imsi},${pred}" }.collect().mkString("\n") )
    bw.close()



    //
    deviceTags.unpersist(true)
  }
}
