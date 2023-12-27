package com.recommend.demo.storagecontrol;

import com.mongodb.MongoClient;
import com.mongodb.MongoClientOptions;
import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import redis.clients.jedis.Jedis;

public class ConnHelper {//建立Mongo 和 Redis 和 连接
    public static RestClient getRestClient(){
        return RestClient.builder(
                new HttpHost("localhost", 9201, "http")
        ).build();
    }
    public static MongoClient getMongoclient(){
        return new MongoClient("localhost",new MongoClientOptions.Builder()
                .maxConnectionIdleTime(30)
                .build());
    }
    public static Jedis getJedisClient(){
        return new Jedis("localhost",6379);
    }

}
