package com.litesnap.open.rwkv;

import android.text.TextUtils;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class StringUtils {
    public static boolean isEmpty(String...strings){
        for (String string : strings){
            if (TextUtils.isEmpty(string)){
                return true;
            }
        }

        return false;
    }

    public static String[] toArrays(String text){
        int[] codePoints = text.codePoints().toArray();
        String[] words = new String[codePoints.length];
        for (int i = 0; i < codePoints.length; i++){
            int code = codePoints[i];
            words[i] = new String(Character.toChars(code));
        }
        return words;
    }

    public static double round(double value, int places) {
        if (places < 0) {
            throw new IllegalArgumentException();
        }
        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
