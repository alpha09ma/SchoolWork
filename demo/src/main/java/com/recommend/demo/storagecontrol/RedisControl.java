package com.recommend.demo.storagecontrol;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.recommend.demo.data.RedisSaveData;
import redis.clients.jedis.Jedis;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RedisControl {
    public static void savetoRedis(RedisSaveData data){
       String json=new Gson().toJson(data);
       Jedis client=ConnHelper.getJedisClient();
//       client.del(data.getId());
       client.lpush(data.getUid(),json);
       client.close();
    }
    public static List<RedisSaveData> loadRedisData(String uid, int num){
        Jedis client=ConnHelper.getJedisClient();
        List<String> result=client.lrange(uid,0,num);
        Function fuc=x->(RedisSaveData)(new Gson().fromJson((String) x,RedisSaveData.class));
        List<RedisSaveData> result1=(List<RedisSaveData>)(result.stream().map(fuc).collect(Collectors.toList()));
        return result1;
    }
}
