package com.litesnap.open.rwkv;

/**
 * Created by ZTMIDGO 2023/7/21
 */
public class HexUtils {
    public static String charsToHex(char[] chars){
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < chars.length; i++) {
            String hex = Integer.toHexString(chars[i]);
            if (hex.length() % 2 != 0) hex = 0 + hex;
            sb.append(hex);
        }
        return sb.toString();
    }
}
