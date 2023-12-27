package com.recommend.demo.controllers.data;

public class RecUserLoginInfo {
    String userid;
    String password;

    public void setUserid(String userid) {
        this.userid = userid;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getUserid() {
        return userid;
    }

    public String getPassword() {
        return password;
    }

    @Override
    public String toString() {
        return "RecUserLoginInfo{" +
                "userid='" + userid + '\'' +
                ", password='" + password + '\'' +
                '}';
    }
}
