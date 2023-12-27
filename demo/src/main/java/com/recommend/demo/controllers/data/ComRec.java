package com.recommend.demo.controllers.data;

public class ComRec<T> {
    private String type;
    private T content;

    public void setType(String type) {
        this.type = type;
    }

    public void setContent(T content) {
        this.content = content;
    }

    public T getContent() {
        return content;
    }

    public String getType() {
        return type;
    }

    @Override
    public String toString() {
        return "ComRec{" +
                "type='" + type + '\'' +
                ", content=" + content +
                '}';
    }
}
