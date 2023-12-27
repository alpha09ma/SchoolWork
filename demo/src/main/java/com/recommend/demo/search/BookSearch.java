package com.recommend.demo.search;

import com.alibaba.fastjson.JSONObject;
import com.recommend.demo.storagecontrol.ConnHelper;
import org.apache.http.HttpHost;
import org.apache.http.util.EntityUtils;
import org.elasticsearch.client.Request;
import org.elasticsearch.client.Response;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestClientBuilder;

import java.io.IOException;

public class BookSearch {
    public static String search(String query) throws IOException {
//        创建客户端
        RestClient restClient = ConnHelper.getRestClient();

        // 构建搜索请求
        Request request = new Request("POST", "/book_info/_search");
        String query_str = "{\"query\":{\"multi_match\":{\"query\":\"" + query + "\",\"fields\":[\"name\",\"author\",\"translator\",\"tag\"]}}}";
        request.setJsonEntity(query_str);
        // 执行搜索请求
        try {
            Response response = restClient.performRequest(request);
            // 处理搜索结果
            String responseBody = EntityUtils.toString(response.getEntity());
            JSONObject responseJson = JSONObject.parseObject(responseBody);
            // 获取_source部分
            String result=responseJson.getString("_source");
            System.out.println("输出：" + result);
            //        关闭连接
            return result;
        } catch (IOException e) {
            // 处理IO异常
            System.out.println(e);
            return null;
        }
    }
}
