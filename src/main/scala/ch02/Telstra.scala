package ch02

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{DataFrame, Row}
/**
 * Created by leorick on 2016/3/11.
 */
object Telstra {
  val appName = ""

  def loadData(sc:SparkContext) = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    //
    case class incident(incid_id:String, locat_id:String, fault_sever_class:String)

    // val incidents = { sc.textFile("file:///home/leoricklin/dataset/telstra/train.csv", 24).
    val incidents = { sc.textFile("file:///home/leo/dataset/telstra/train.csv", 24).
      map{ line => line.split(""",""")}.filter{ case ary => !(ary(0).equals("id")) } }.
      map{ ary => incident(ary(0), ary(1), ary(2))}.toDF()
    incidents.count // = #trainset = 7381
    // DF.cache 將導致 join 結果異常
    /*
    incidents.take(5) = Array(
[14121,location 118,1],
[9320,location 91,0],
[14394,location 152,1],
[8218,location 931,1],
[14804,location 120,0])
     */
    //
    case class severity(se_incid_id:String, sever_type_id:String)
    // val severities = sc.textFile("file:///home/leoricklin/dataset/telstra/severity_type.csv", 24).
    val severities = sc.textFile("file:///home/leo/dataset/telstra/severity_type.csv", 24).
      map{ line => line.split(""",""")}.filter{ case ary => !(ary(0).equals("id")) }.
      map{ ary => severity(ary(0), ary(1)) }.toDF()
    severities.count() // = 18552
    severities.select("sever_type_id").distinct().count() // #unique_severity_type = 5
    val r01: DataFrame = incidents.join(severities, incidents("incid_id") === severities("se_incid_id"))
    r01.groupBy(incidents("incid_id")).count(). // = [incid_id: string, count: bigint]
      where(r01("count") > 1).count // incident:severity_type = 1:1
    /*
    severities.take(5) = Array(
[6597,severity_type 2],
[8011,severity_type 2],
[2597,severity_type 2],
[5022,severity_type 1],
[6852,severity_type 1])
    severities.groupBy("sever_type_id").count.collect  = Array(
[severity_type 1,8728],
[severity_type 2,8737],
[severity_type 3,8],
[severity_type 4,1014],
[severity_type 5,65])
     */
    //
    case class event(ev_incid_id:String, event_type_id:String)
    // val events = sc.textFile("file:///home/leoricklin/dataset/telstra/event_type.csv", 24).
    val events = sc.textFile("file:///home/leo/dataset/telstra/event_type.csv", 24).
      map{ line => line.split(""",""")}.filter{ case ary => !(ary(0).equals("id")) }.
      map{ ary => event(ary(0), ary(1))}.toDF()
    events.count // = 31170
    events.select("event_type_id").distinct().count() // #unique_event_type = 53
    val r02 = incidents.join(events, incidents("incid_id") === events("ev_incid_id"))
    r02.groupBy(incidents("incid_id")).count. // = [incid_id: string, count: bigint]
      where(r02("count") > 1).count() // incident:event_type = 1:N
    /*
    events.take(5) = Array(
[6597,event_type 11],
[8011,event_type 15],
[2597,event_type 15],
[5022,event_type 15],
[5022,event_type 11])
     */
    //
    case class resource(re_incid_id:String, resource_type_id:String)
    // val resources = sc.textFile("file:///home/leoricklin/dataset/telstra/resource_type.csv", 24).
    val resources = sc.textFile("file:///home/leo/dataset/telstra/resource_type.csv", 24).
      map{ line => line.split(""",""")}.filter{ case ary => !(ary(0).equals("id")) }.
      map{ ary => resource(ary(0), ary(1))}.toDF()
    resources.select("resource_type_id").distinct().count // #unique_resource_type = 10
    val r03 = incidents.join(resources, incidents("incid_id") === resources("re_incid_id"))
    r03.groupBy(incidents("incid_id")).count(). // = [incid_id: string, count: bigint]
      where(r03("count") > 1).count // incident:resource_type = 1:N
    //
    case class feature(fe_incid_id:String, feature_id:String, volume:Double)
    // val features = sc.textFile("file:///home/leoricklin/dataset/telstra/log_feature.csv", 24).
    val features = sc.textFile("file:///home/leo/dataset/telstra/log_feature.csv", 24).
      map{ line => line.split(""",""")}.filter{ case ary => !(ary(0).equals("id")) }.
      map{ ary => feature(ary(0), ary(1), ary(2).toDouble)}.toDF()
    features.select("feature_id").distinct().count() // #unique_feature = 386
    val r04 = incidents.join(features, incidents("incid_id") === features("fe_incid_id"))
    r04.groupBy(incidents("incid_id")).count(). // = [incid_id: string, count: bigint]
      where(r04("count") > 1).count() // incident:feature = 1:N
  }

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName(appName)
    val sc = new SparkContext(sparkConf)

  }
}
