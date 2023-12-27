package com.recommend.demo.controllers.data;

public class ComRespond {
    private int type;
    private String respond;
    private String msg;
    public ComRespond(int type,String respond,String msg){
        this.type=type;
        this.msg=msg;
        this.respond=respond;
    }
    public void setType(int type) {
        this.type = type;
    }

    public void setMsg(String msg) {
        this.msg = msg;
    }

    public void setRespond(String respond) {
        this.respond = respond;
    }

    public int getType() {
        return type;
    }

    public String getMsg() {
        return msg;
    }

    public String getRespond() {
        return respond;
    }
}
