package com.recommend.demo.realtimerecommend;

import org.apache.spark.internal.config.Python;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.io.UnsupportedEncodingException;

public class Recommend {
    public void recommend(String userid) throws UnsupportedEncodingException {
        predict(userid);
        System.out.println(1111);
    }
    public void predict(String userid) throws UnsupportedEncodingException {//情感分析评论
        try {
            Process proc=Runtime.getRuntime().exec("python3 "+"。/python/graph_recommend.py"+" "+userid); //执行py文件
            InputStreamReader stdin=new InputStreamReader(proc.getInputStream());
            LineNumberReader input=new LineNumberReader(stdin);
            String line;
            while((line=input.readLine())!=null ){
                System.out.println(line);//得到输出
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
