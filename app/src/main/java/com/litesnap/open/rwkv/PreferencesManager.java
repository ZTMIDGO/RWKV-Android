package com.litesnap.open.rwkv;

/**
 * Created by ZTMIDGO 2023/6/20
 */
public class PreferencesManager {
    public static int getTopK(){
        return PreferencesUtils.getInt(Atts.TOP_K, 0);
    }

    public static int getLen(){
        return PreferencesUtils.getInt(Atts.LEN, 512);
    }

    public static float getP1(){
        return PreferencesUtils.getFloat(Atts.P1, 0.4f);
    }

    public static float getP2(){
        return PreferencesUtils.getFloat(Atts.P2, 0.4f);
    }

    public static float getTemp(){
        return PreferencesUtils.getFloat(Atts.TEMP, 1.0f);
    }

    public static float getTopp(){
        return PreferencesUtils.getFloat(Atts.TOP_P, 0.1f);
    }
}
