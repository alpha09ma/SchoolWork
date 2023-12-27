package com.recommend.demo.labelrecommend;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mongodb.MongoClient;
import com.mongodb.MongoClientOptions;
import com.mongodb.MongoClientURI;
import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Sorts;
import com.recommend.demo.storagecontrol.ConnHelper;
import org.bson.BsonArray;
import org.bson.Document;

import java.util.ArrayList;


public class LabelRecommend {

    public static String recommend(String user_id) {
//        连接Mongodb


        MongoDatabase database;
        MongoClient client = ConnHelper.getMongoclient();
//        连接到Mongodb数据库
        database = client.getDatabase("recommend");
        System.out.println("成功连接到数据库");


//        获取用户信息集合
        MongoCollection<Document> user_collection = database.getCollection("userinfo");
//        查询用户喜好的图书标签
        Document user_query = new Document("userId", user_id);
        FindIterable<Document> user_results = user_collection.find(user_query);
        String tags_str;
        String[] tags;
        Document user_info;
        System.out.println(user_results.first());
        StringBuilder recommended_books = new StringBuilder();
        JsonArray array=new JsonArray();
        user_info=user_results.first();
        tags_str = user_info.getString("tags");
        if (tags_str!=null) {
            tags = tags_str.split("/");
//        获取用户已经看过的书籍的bookId
            ArrayList<Integer> viewed_books = (ArrayList<Integer>) user_info.get("viewedBooks");
//        获取书籍信息集合
            MongoCollection<Document> books_collection = database.getCollection("bookinfo");
//        根据标签依次查找书籍

            recommended_books.append("[");
            for (String tag : tags) {
                Document query = new Document();
                query.append("tag", new Document("$regex", tag));
//            排除已经看过的书籍
                if (viewed_books != null)
                    query.append("bookId", new Document("$nin", viewed_books));
                else
                    query.append("bookId", new Document("$nin", new ArrayList<String>()));
//            查找并按分数和打分人数排序
                FindIterable<Document> book_results = books_collection.find(query)
                        .sort(Sorts.descending("rating", "ratingNum")).limit(10);
                try (MongoCursor<Document> cursor = book_results.iterator()) {
//            每个标签推荐10部电影
                    while (cursor.hasNext()) {
                        Document book = cursor.next();
                        book.remove("comments");
                        array.add(book.toJson());
                        recommended_books.append(book.toJson()+",");
                    }
                }
            }
            recommended_books.insert(recommended_books.lastIndexOf(","),"");
            recommended_books.append("]");
        }
        return array.toString().replaceAll("NaN","null");
    }
    public static String getBookInfo(String book_id){

        MongoDatabase database;
        MongoClient client = ConnHelper.getMongoclient();
//        连接到Mongodb数据库
        database = client.getDatabase("recommend");
        System.out.println("成功连接到数据库");

//        获取书籍信息集合
        MongoCollection<Document> book_collection = database.getCollection("bookinfo");
        Document book_query = new Document("bookId", Integer.parseInt(book_id));
        Document bookinfo=book_collection.find(book_query).first();
        if(bookinfo!=null)
            return bookinfo.toJson().replaceAll("NaN","null");
        else
            return "";
    }
    public static String getRecommend(String user_id){
        MongoDatabase database;
        MongoClient client = ConnHelper.getMongoclient();
//        连接到Mongodb数据库
        database = client.getDatabase("recommend");
        System.out.println("成功连接到数据库");
        MongoCollection<Document> rec_collection = database.getCollection("movie_rec");
        Document recinfo=rec_collection.find(new Document("userid",user_id)).first();
        JsonArray array=new JsonArray();
        if(recinfo!=null)
        {
        ArrayList<Integer> rec_id= (ArrayList<Integer>) recinfo.get("recs");
//        获取书籍信息集合
        MongoCollection<Document> book_collection = database.getCollection("bookinfo");
        Document book_query = new Document();
        book_query.append("bookId",new Document("$in",rec_id));
        FindIterable<Document> bookinfo=book_collection.find(book_query);
        for (Document book:bookinfo)
        {
            book.remove("comments");
            array.add(book.toJson());
        }

        if(bookinfo!=null)
            return array.toString().replaceAll("NaN","null");
        else
            return "";
        }
        else
            return "0";
    }
}
