package com.recommend.demo;

import com.recommend.demo.realtimerecommend.StreamingRecommender;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration;
import org.springframework.boot.autoconfigure.data.redis.RedisRepositoriesAutoConfiguration;
import org.springframework.boot.autoconfigure.mongo.MongoAutoConfiguration;


@SpringBootApplication(exclude = {
//        禁止mongodb自动装配
		MongoAutoConfiguration.class,
//        禁止redis自动装配
		RedisAutoConfiguration.class,
		RedisRepositoriesAutoConfiguration.class,
})
public class ProjectRecommendDemoApplication {

	public static void main(String[] args) throws InterruptedException {
//		SparkStreaming sparkStreaming=new SparkStreaming();
		SpringApplication.run(ProjectRecommendDemoApplication.class, args);
		StreamingRecommender.main(args);

//		sparkStreaming.startRecentlyRecommend();//kidstarlets
	}

}
