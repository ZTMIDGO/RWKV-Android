package com.litesnap.open.rwkv;

/**
 * Created by ZTMIDGO 2023/6/20
 */
public class PreferencesManager {
    public static int getTopK(){
        return PreferencesUtils.getInt(Atts.TOP_K, 1);
    }

    public static int getLen(){
        return PreferencesUtils.getInt(Atts.LEN, 512);
    }

    public static float getP1(){
        return PreferencesUtils.getFloat(Atts.P1, 0.7f);
    }

    public static float getP2(){
        return PreferencesUtils.getFloat(Atts.P2, 0.4f);
    }
}
