package com.litesnap.open.rwkv;

import java.util.ArrayList;
import java.util.List;

public class MCUUtils {

    public static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bytes.length; i++) {
            String hex = byteToHex(bytes[i]);
            sb.append(hex);
        }
        return sb.toString();
    }

    public static String byteToHex(byte b){
        String hex = Integer.toHexString(b & 0xFF);
        int patch = hex.length() % 2;
        StringBuilder sb = new StringBuilder();
        if (patch != 0){
            for (int i = 0; i < patch; i++){
                sb.append("0");
            }
        }
        sb.append(hex);
        return sb.toString();
    }
}
