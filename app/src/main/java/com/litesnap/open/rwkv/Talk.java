package com.litesnap.open.rwkv;

/**
 * Created by ZTMIDGO 2023/6/20
 */
public class Talk {
    public static final int TYPE_QUESTION = 0;
    public static final int TYPE_ANSWER = 1;

    private int type;
    private String text;

    public Talk(int type, String text) {
        this.type = type;
        this.text = text;
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }
}
