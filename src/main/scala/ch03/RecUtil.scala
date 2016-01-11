package ch03

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
 * Created by leorick on 2015/9/4.
 */
object RecUtil extends Serializable {
  def areaUnderCurve(positiveData: RDD[Rating], bAllItemIDs: Broadcast[Array[Int]]
  , predictFunction: (RDD[(Int,Int)] => RDD[Rating]) ): Double = {
    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive", and map to tuples
    val positiveUserProducts: RDD[(Int, Int)] = positiveData.map(r => (r.user, r.product))
    // Make predictions for each of them, including a numeric score, and gather by user
    val positivePredictions: RDD[(Int, Iterable[Rating])] = predictFunction(positiveUserProducts).groupBy(_.user)

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other items, excluding those that are "positive" for the user.
    // 需要再確認: 負例資料 = 所有 artist 與 user-played-artist (in cv dataset) 的差集
    // , 其結果可能包含 user-played-artist (in train set)
    val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions {
      // mapPartitions operates on many (user,positive-items) pairs at once
      userIDAndPosItemIDs: Iterator[(Int, Iterable[Int])] => {
        // Init an RNG and the item IDs set once for partition
        val random = new Random()
        val allItemIDs: Array[Int] = bAllItemIDs.value
        // 負例清單=所有 artist 與 user-played-artist 的差集, 數量=該用戶已 rating 的 artists 數量, 負例清單的 artist 可重複
        userIDAndPosItemIDs.map { case (userID: Int, posItemIDs: Iterable[Int]) =>
          val posItemIDSet = posItemIDs.toSet
          val negative = new ArrayBuffer[Int]()
          var i = 0
          // Keep about as many negative examples per user as positive.
          // Duplicates are OK
          while (i < allItemIDs.size && negative.size < posItemIDSet.size) {
            val itemID = allItemIDs(random.nextInt(allItemIDs.size))
            if (!posItemIDSet.contains(itemID)) {
              negative += itemID
            }
            i += 1
          }
          negative.map(itemID => (userID, itemID)) // collection of (user,negative-item) tuples
        }
      }
    }.flatMap(t => t)
    // flatMap breaks the collections above down into one big set of tuples

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user)

    // Join positive and negative by user
    positivePredictions.join(negativePredictions).values.map {
      case (positiveRatings, negativeRatings) =>
        // AUC may be viewed as the probability that a random positive item scores
        // higher than a random negative one. Here the proportion of all positive-negative
        // pairs that are correctly ranked is computed. The result is equal to the AUC metric.
        var correct = 0L
        var total = 0L
        // For each pairing,
        for (positive <- positiveRatings;
             negative <- negativeRatings) {
          // Count the correctly-ranked pairs
          if (positive.rating > negative.rating) {
            correct += 1
          }
          total += 1
        }
        // Return AUC: fraction of pairs ranked correctly
        correct.toDouble / total
    }.mean() // Return mean AUC over users
  }

  def buildRatings(rawUserArtistData: RDD[String]
   ,bArtistAlias: Broadcast[scala.collection.Map[Int,Int]]) : RDD[Rating] = {
    rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)     // tokenize
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)  // clean & transform to unique artist ID
      Rating(userID, finalArtistID, count)
    }
  }

  def buildArtistAlias(rawArtAli: RDD[String]): scala.collection.Map[Int,Int] =
    rawArtAli.flatMap { line =>
      val tokens = line.split('\t') // tokenize
      if (tokens.size != 2 || tokens(0).isEmpty || tokens(1).isEmpty) {
        None
      } else { // clean & transform to Int type
        Some((tokens(0).toInt, tokens(1).toInt))
      }
    }.collectAsMap()


  def buildArtistByID(rawArtistData: RDD[String]): RDD[(Int, String)] = {
    rawArtistData.flatMap { line =>
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
  }

  def predictMostListened(sc: SparkContext, train: RDD[Rating])(allData: RDD[(Int,Int)]): RDD[Rating] = {
    val bListenCount = sc.broadcast(
      train.map(r => (r.product, r.rating)).reduceByKey(_ + _).collectAsMap()
    )
    allData.map { case (user, product) =>
      Rating(user, product, bListenCount.value.getOrElse(product, 0.0) )
    }
  }
}
