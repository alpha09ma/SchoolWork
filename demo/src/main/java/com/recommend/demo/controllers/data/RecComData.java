package com.recommend.demo.controllers.data;

public class RecComData {
    String userid;
    String bookid;
    Double rating;
    String comment;

    public void setUserid(String userid) {
        this.userid = userid;
    }

    public void setComment(String comment) {
        this.comment = comment;
    }

    public void setRating(Double rating) {
        this.rating = rating;
    }

    public void setBookid(String bookid) {
        this.bookid = bookid;
    }

    public String getUserid() {
        return userid;
    }

    public String getComment() {
        return comment;
    }

    public Double getRating() {
        return rating;
    }

    public String getBookid() {
        return bookid;
    }

    @Override
    public String toString() {
        return "RecComData{" +
                "userid='" + userid + '\'' +
                ", bookid='" + bookid + '\'' +
                ", rating=" + rating +
                ", comment='" + comment + '\'' +
                '}';
    }
}
