package com.recommend.demo.controllers.data;

public class RecUserLabel {
    private String userid;
    private String label;

    public void setUserid(String userid) {
        this.userid = userid;
    }

    public String getUserid() {
        return userid;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "RecUserLabel{" +
                "userid='" + userid + '\'' +
                ", label='" + label + '\'' +
                '}';
    }
}
