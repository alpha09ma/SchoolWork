package com.recommend.demo.realtimerecommend

import breeze.storage.ConfigurableDefault.fromV
import com.recommend.demo.data.RedisSaveData
import com.recommend.demo.storagecontrol.{ConnHelper, RedisControl}
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.kafka010.{ConsumerStrategies, KafkaUtils, LocationStrategies}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.bson.Document
import redis.clients.jedis.Jedis

import java.util
import scala.collection.JavaConverters._
import scala.collection.convert.ImplicitConversions.{`collection AsScalaIterable`, `iterable AsScalaIterable`, `map AsJavaMap`}
import scala.collection.mutable.ArrayBuffer

case class MongConfig(uri:String,db:String)
// 标准推荐
case class Recommendation(mid:String, score:java.lang.Double)
// 用户的推荐
case class UserRecs(uid:String, recs:Seq[Recommendation])
//电影的相似度
case class MovieRecs(mid:String, recs:Seq[Recommendation])

object StreamingRecommender {
    val MAX_USER_RATINGS_NUM = 20
    val MAX_SIM_MOVIES_NUM = 20
    val MONGODB_STREAM_RECS_COLLECTION = "StreamRecs"
    val MONGODB_RATING_COLLECTION = "Rating"
    val MONGODB_MOVIE_RECS_COLLECTION = "movie_sim"
    //入口方法
    def main(args: Array[String]): Unit = {
      val config = Map(
        "spark.cores" -> "local[2]",
        "mongo.uri" -> "mongodb://localhost:27017/recommend",
        "mongo.db" -> "recommend",
        "kafka.topic" -> "recommend"
      )
      //创建一个 SparkConf 配置
      val sparkConf = new
          SparkConf().setAppName("StreamingRecommender").setMaster(config("spark.cores"))
      sparkConf.set("spark.master","local")
      sparkConf.set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
      val spark = SparkSession.builder().config(sparkConf).getOrCreate()
      val sc = spark.sparkContext
      val ssc = new StreamingContext(sc, Seconds(2))
      implicit val mongConfig = MongConfig(config("mongo.uri"), config("mongo.db"))
//      import spark.implicits._
//      val simMoviesMatrix = spark
//        .read
//        .option("uri", config("mongo.uri"))
//        .option("collection", MONGODB_MOVIE_RECS_COLLECTION)
//        .format("com.mongodb.spark.sql")
//        .load()
//        .as[MovieRecs]
//        .rdd
//        .map { recs =>
//          (recs.mid, recs.recs.map(x => (x.mid, x.score)).toMap)
//        }.collectAsMap()
//      println(simMoviesMatrix)
      //创建到 Kafka 的连接
      val kafkaPara = Map(
        "bootstrap.servers" -> "localhost:9092",
        "key.deserializer" -> classOf[StringDeserializer],
        "value.deserializer" -> classOf[StringDeserializer],
        "group.id" -> "recommender",
        "auto.offset.reset" -> "latest"
      )
      val kafkaStream =
        KafkaUtils.createDirectStream[String, String](ssc, LocationStrategies.PreferConsistent, ConsumerStrategies.Subscribe[String, String](Array(config("kafka.topic")), kafkaPara))
      // UID|MID|SCORE|TIMESTAMP
      // 产生评分流
      val ratingStream = kafkaStream.map { case msg =>
        var attr = msg.value().split(",")
        (attr(0), attr(1), attr(2).toDouble, attr(3).toLong)
      }
      // 核心实时推荐算法
      ratingStream.foreachRDD { rdd =>
        rdd.map { case (uid, mid, score, timestamp) =>
          println(">>>>>>>>>>>>>>>>")
          //获取当前最近的 M 次电影评分
          val userRecentlyRatings = RedisControl.loadRedisData(uid,MAX_USER_RATINGS_NUM).toList.map{
            x  => (x.getId,x.getRating.toDouble)
            }.toArray
          println(userRecentlyRatings.length)
          //获取电影 P 最相似的 K 个电影
          val simMovies =
            getTopSimMovies(MAX_SIM_MOVIES_NUM, mid, uid)
          println("topsim"+simMovies(4))
          //计算待选电影的推荐优先级
          val l=simMovies.toBuffer
          l.drop(0)

          val streamRecs = computeMovieScores(userRecentlyRatings, l.toArray)
//          //将数据保存到 MongoDB
//          val streamRecs1=streamRecs.toBuffer
//          streamRecs1.drop(0)
          saveRecsToMongoDB(uid, streamRecs.toArray)
//          println("result"+streamRecs.length)
        }.count()
      }
      ssc.start()
      ssc.awaitTermination()
    }
  def saveRecsToMongoDB(uid:String,streamRecs:Array[(String,Double)])(implicit mongConfig:
  MongConfig): Unit ={
    //到 StreamRecs 的连接
    val streaRecsCollection =
      ConnHelper.getMongoclient.getDatabase("recommend").getCollection("movie_rec")
    streaRecsCollection.findOneAndDelete(new Document("uid",uid))
    val data=new Document()
    val arrayBuffer=ArrayBuffer[Integer]()
    for ((i:String,j:Double)<-streamRecs)
        arrayBuffer.append(i.toInt)

    data.append("userid",uid)
    data.append("recs",arrayBuffer.toArray.toList.asJava)
    streaRecsCollection.insertOne(data)
  }
  def loadsimmovies(mid:String): List[(String,Double)]=
  {
    val client=ConnHelper.getMongoclient()
    val db=client.getDatabase("recommend")
    val collection=db.getCollection("movie_sim")
    val document=new Document()
    val result=collection.find(document).first().get("recs") match {
      case list:java.util.ArrayList[AnyRef]=>list.toList
    }
    val result1=result.map{
      case x:Document=> (x.get("mid") match {
        case y:java.lang.Integer=>y.toString
      },x.get("score") match {
        case z:java.lang.Double=>z.toDouble
      })
    }
    return result1
  }
  def getTopSimMovies(num:Int, mid:String, uid:String)(implicit mongConfig: MongConfig): Array[String] ={
    //从广播变量的电影相似度矩阵中获取当前电影所有的相似电影

    val allSimMovies = loadsimmovies(mid).toArray
    //获取用户已经观看过得电影
    val document=new Document()
    document.append("uid",uid)
    val ratingExist =
      ConnHelper.getMongoclient().getDatabase("recommend").getCollection("userinfo").find(document).toArray.map{
        item =>item.get("viewedBooks").toString
      }
    //过滤掉已经评分过得电影，并排序输出
    allSimMovies.filter(x => !ratingExist.contains(x._1)).sortWith(_._2 >
      _._2).take(num).map(x => x._1.toString)
  }
  def computeMovieScores(userRecentlyRatings:Array[(String,Double)],topSimMovies: Array[String]):
  Array[(String,Double)] ={
    //用于保存每一个待选电影和最近评分的每一个电影的权重得分
    val score = scala.collection.mutable.ArrayBuffer[(String,Double)]()
    //用于保存每一个电影的增强因子数
    val increMap = scala.collection.mutable.HashMap[String,Int]()
    //用于保存每一个电影的减弱因子数
    val decreMap = scala.collection.mutable.HashMap[String,Int]()
    for (topSimMovie <- topSimMovies; userRecentlyRating <- userRecentlyRatings){
      val simScore = getMoviesSimScore(userRecentlyRating._1,topSimMovie)
      score += ((topSimMovie, simScore * userRecentlyRating._2 ))
      if(userRecentlyRating._2 > 3){
        increMap(topSimMovie) = increMap.getOrDefault(topSimMovie,0) + 1
      }else{
        decreMap(topSimMovie) = decreMap.getOrDefault(topSimMovie,0) + 1
      }

    }

    score.groupBy(_._1).map{case (mid,sims) =>
      (mid,sims.map(_._2).sum / sims.length + log(increMap.getOrDefault(mid, 1)) -
        log(decreMap.getOrDefault(mid, 1)))
    }.toArray.sortWith(_._2>_._2)
  }
  def getMoviesSimScore(userRatingMovie:String, topSimMovie:String): Double ={
    val simMovie=loadsimmovies(topSimMovie).toMap
      simMovie.get(userRatingMovie) match {
      case Some (score) => score
      case None => 0.0
      }
  }
  def log(m:Int):Double ={
    math.log(m) / math.log(10)
  }


}
