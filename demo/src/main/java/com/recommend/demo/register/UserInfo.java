package com.recommend.demo.register;

import com.mongodb.MongoClient;
import com.mongodb.MongoException;
import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.recommend.demo.storagecontrol.ConnHelper;
import org.bson.Document;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Random;

public class UserInfo {
    /*将用户注册信息写入数据库*/
    static MongoClient client= ConnHelper.getMongoclient();
    /*注册时写入基本信息：userName,password，并返回userId*/
    public static String storeBasicInfo(String user_name,String password){

        //        连接到Mongodb数据库
        MongoDatabase database = client.getDatabase("recommend");
        System.out.println("成功连接到数据库");
        MongoCollection<Document> user_collection = database.getCollection("userinfo");
//        生成user_id
        String user_id;
        while (true){
            Random ran=new Random();
            Integer id=ran.nextInt(899999)+100000;
            user_id=id.toString();
//             检查该id是否已经存在
            Document query = new Document("userId", user_id);
            long count=user_collection.countDocuments(query);
            if(count==0)
                break;
        }
        Document basic=new Document();
        basic.append("userId",user_id);
        basic.append("userName",user_name);
        basic.append("password",password);
        try {
            user_collection.insertOne(basic);
            System.out.println("用户信息插入成功！\n");
            return user_id;
        } catch (MongoException e) {
            System.out.println("用户信息插入失败：" + e.getMessage());
            return "";
        }
    }

    /*登录验证*/
    public static boolean login(String user_id,String password){
        MongoDatabase database;
            //        连接到Mongodb数据库
        database = client.getDatabase("recommend");
        MongoCollection<Document> user_collection=database.getCollection("userinfo");
        Document query=new Document("userId",user_id);
        Document user_doc=user_collection.find(query).first();
        String right_password=user_doc.getString("password");
        if(password.equals(right_password))
            return true;
        else return false;
    }

    /*写入用户喜爱的标签信息*/
    public static boolean storeLikedTags(String user_id,String tags){
        MongoDatabase database;
        database = client.getDatabase("recommend");
        System.out.println("成功连接到数据库");
        MongoCollection<Document> user_collection=database.getCollection("userinfo");
        Document query=new Document("userId",user_id);
        Document tagfield=new Document("tags",tags);
        Document update=new Document("$set",tagfield);
        try{
            user_collection.updateOne(query,update);
            System.out.println("用户喜爱标签添加成功！\n");
            return true;
        }
        catch (MongoException e){
            System.out.println("用户喜爱标签添加失败：" + e.getMessage());
            return false;
        }
    }

    /*写入看过的书籍（users）和打分信息（bookinfo）*/
    public boolean storeRating(String user_id,int book_id,int rating,String comment){
        //        连接到Mongodb数据库
        MongoDatabase database = client.getDatabase("recommend");
        System.out.println("成功连接到数据库");
        //        将用户看过的书加入到用户信息表中
        MongoCollection<Document> user_collection=database.getCollection("userinfo");
        Document query=new Document("userId",user_id);
        Document addBook=new Document("$push",new Document("viewedBooks",book_id));
        try{
            user_collection.updateOne(query,addBook);
            System.out.println("用户信息表更新成功！\n");
            return true;
        }
        catch (MongoException e){
            System.out.println("用户信息表更新失败：" + e.getMessage());
            return false;
        }

        //        将用户打分和评论写入图书信息表中
//        MongoCollection<Document> book_collection=database.getCollection("bookinfo");
//        Document book_query=new Document("bookId",book_id);
//        Document ratinginfo=new Document();
//        ratinginfo.append("userId",user_id);
//        Document user_info=user_collection.find(query).first();
//        String user_name=user_info.getString("userName");
//        ratinginfo.append("userName",user_name);
//        ratinginfo.append("bookId",book_id);
//        ratinginfo.append("rating",rating);
//        Date date = new Date();
//        //        可能需要改成String类型?
//        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
//        ratinginfo.append("time", formatter.format(date));
//        ratinginfo.append("content",comment);
//        try{
//            book_collection.updateOne(book_query,new Document("$push",new Document("comments",ratinginfo)));
//            System.out.println("书籍信息表添加评论成功！\n");
//        }
//        catch (MongoException e){
//            System.out.println("书籍信息表添加评论失败：" + e.getMessage());
//        }
//
//        //        更新评分人数
//
//        //        更新评分
    }

    /*获取用户的所有书籍打分，返回书籍id*/
    public String getRatingBook(String user_id){
        MongoDatabase database;
        database = client.getDatabase("recommend");
        System.out.println("成功连接到数据库");
        MongoCollection<Document> user_collection=database.getCollection("userinfo");
        Document query=new Document("userId",user_id);
        Document result=user_collection.find(query).first();
//        return (ArrayList<Integer>) (result != null ? result.get("viewedBooks") : null);
        ArrayList<Integer> book_id_list=new ArrayList<>();
        if(result!=null){
            book_id_list=(ArrayList<Integer>) result.get("viewedBooks");
        }
        else {
            return null;
        }
        MongoCollection<Document> book_collection = database.getCollection("bookinfo");
        StringBuilder ratings_str=new StringBuilder();
        for(Integer id:book_id_list){
            Document rating_query=new Document();
            rating_query.append("bookId",id);
            Document book_doc=book_collection.find(rating_query).first();
            assert book_doc != null;
            String book_name=book_doc.getString("bookName");
            String book_url=book_doc.getString("imgUrl");
            rating_query.append("comments",new Document("$elemMatch",new Document("userId",user_id)));
            FindIterable<Document> rating_results = user_collection.find(rating_query);
            for (Document rating:rating_results){
                String rating_str="{\"bookName\":"+book_name+",\"imgUrl\":"+book_url+",\"bookId\":"+id+",\"rating\":"+rating.getString("rating")+
                        ",\"time\":"+rating.getString("time")+",\"comment\":"+rating.getString("content")+"}";
                ratings_str.append(rating_str).append("\n");
            }
        }
        return ratings_str.toString();
    }

}
