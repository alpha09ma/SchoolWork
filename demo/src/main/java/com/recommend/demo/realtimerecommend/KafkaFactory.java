package com.recommend.demo.realtimerecommend;

import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.producer.KafkaProducer;

import java.io.Closeable;
import java.util.Properties;

public class KafkaFactory {
    public static Closeable get(int type,Properties properties){
        if (type==0)
        return new KafkaConsumer<String,String>(properties);
        else
            return new KafkaProducer<String,String>(properties);
    }
}
