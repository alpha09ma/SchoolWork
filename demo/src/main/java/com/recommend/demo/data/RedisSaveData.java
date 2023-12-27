package com.recommend.demo.data;

public class RedisSaveData {
    transient String uid;
    String id;
    Double rating;
    String comment;
    Long timestamp;
    public RedisSaveData(String uid,String id,Double rating,String comment,Long timestamp){
        this.comment=comment;
        this.id=id;
        this.rating=rating;
        this.uid=uid;
        this.timestamp=timestamp;
    }
    public void setId(String id) {
        this.id = id;
    }

    public void setRating(Double rating) {
        this.rating = rating;
    }

    public void setUid(String uid) {
        this.uid = uid;
    }

    public String getUid() {
        return uid;
    }

    public void setComment(String comment) {
        this.comment = comment;
    }

    public String getId() {
        return id;
    }

    public Double getRating() {
        return rating;
    }

    public String getComment() {
        return comment;
    }

    public Long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Long timestamp) {
        this.timestamp = timestamp;
    }

    @Override
    public String toString() {
        return "RedisSaveData{" +
                "id='" + id + '\'' +
                ", rating='" + rating + '\'' +
                ", comment='" + comment + '\'' +
                ", timestamp=" + timestamp +
                '}';
    }
}
