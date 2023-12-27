package com.recommend.demo.controllers;

import com.recommend.demo.controllers.data.*;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.List;
import java.util.Properties;

import com.recommend.demo.data.RedisSaveData;
import com.recommend.demo.labelrecommend.LabelRecommend;
import com.recommend.demo.realtimerecommend.KafkaFactory;
import com.recommend.demo.register.UserInfo;
import com.recommend.demo.search.BookSearch;
import com.recommend.demo.storagecontrol.RedisControl;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.kafka.common.serialization.StringSerializer;
import org.springframework.web.bind.annotation.*;

@RestController
@CrossOrigin
public class Controller {
    @PostMapping("/register")
    public ComRespond registerAccount(@RequestBody RecUserInfo msg){
        String id=UserInfo.storeBasicInfo(msg.getUsername(),msg.getPassword());
        return new ComRespond(0,"ok",id);
    }
    @PostMapping("/labelselect")
    public ComRespond userlabelSelect(@RequestBody RecUserLabel msg){
        System.out.println(msg);
        if(UserInfo.storeLikedTags(msg.getUserid(),msg.getLabel()))
            return new ComRespond(1,"ok",null);
        else
            return new ComRespond(1,"false",null);
    }
    @PostMapping("/login")
    public ComRespond login(@RequestBody RecUserLoginInfo msg){
        if(UserInfo.login(msg.getUserid(),msg.getPassword()))
            return new ComRespond(2,"ok","");
        else
            return new ComRespond(2,"false","");
    }
    @GetMapping("/search")
    public String searchRecommend(@RequestParam("query") String msg) throws IOException {
        String test=BookSearch.search(msg);
        return test;
    }
    @GetMapping("/label")
    public String labelRecommend(@RequestParam("userid") String msg){
        System.out.println(msg);
        return LabelRecommend.recommend(msg);
    }
//    @GetMapping("/history")
//    public String history(@RequestParam("id") String msg){
//        return UserInfo.getRatingBook(msg);
//    }
    @GetMapping("/bookinfo")
    public String bookinfo(@RequestParam("bookid") String msg){
        return LabelRecommend.getBookInfo(msg);
    }
    @PostMapping("/comment")
    public ComRespond saveComment(@RequestBody RecComData msg) throws UnsupportedEncodingException {
        RedisControl.savetoRedis(new RedisSaveData(msg.getUserid(),"26362836",msg.getRating(),msg.getComment(),System.currentTimeMillis()));
        //List comment=RedisControl.loadRedisData(msg.getId(),1);
        Properties properties=new Properties();
        properties.put( "bootstrap.servers","localhost:9092");
        //properties.put( "key.deserializer", StringDeserializer.class);
        //properties.put( "value.deserializer",StringDeserializer.class);
        properties.put("key.serializer", StringSerializer.class);
        properties.put("value.serializer",StringSerializer.class);
        properties.put( "retries","0");
        properties.put("acks", "all");
        KafkaProducer<String,String> kafkaProducer=(KafkaProducer<String, String>) KafkaFactory.get(1,properties);
        kafkaProducer.send(new ProducerRecord<String, String>("recommend","predict",msg.getUserid()+","+msg.getBookid()+","+msg.getRating()+","+System.currentTimeMillis()));
        return new ComRespond(3,"ok","");
    }
    @GetMapping("/home")
    public String realtimeRecommend(@RequestParam("userid") String userid){
        String result=LabelRecommend.getRecommend(userid);
        //List comment=RedisControl.loadRedisData(msg.getId(),1);
        if (result.equals("0"))
            return LabelRecommend.recommend(userid);
        else
            return result;
    }
}
