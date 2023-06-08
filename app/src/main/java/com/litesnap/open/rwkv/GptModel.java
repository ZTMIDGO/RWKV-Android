package com.litesnap.open.rwkv;

import java.util.List;

/**
 * Created by ZTMIDGO 2022/9/9
 */
public interface GptModel {
    void generate(String text, int maxCount, Callback callback);
    int sample(List<Integer> indexes, List<Float> probs);
    void close();
    void cancel();
    void setTopK(int value);
    interface Callback{
        void callback(String text);
    }
}
