package com.litesnap.open.rwkv;

import android.content.Context;
import android.util.Log;

import com.google.gson.Gson;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.MatchResult;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Created by ZTMIDGO 2022/9/15
 */
public class WorldTokenizerImp implements GptTokenizer {
    private final String VOCAB_NAME = "vocab.json";
    private final Map<String, Integer> encoder = new HashMap<>();
    private final Map<Integer, String> decoder = new HashMap<>();
    private final Set<String> tiesSet = new HashSet<>();
    private final Context context;

    public WorldTokenizerImp(Context context){
        this.context = context;
        fillDecoder();
        fillEncoder();
    }

    @Override
    public List<Integer> encode(String text) {
        List<Integer> result = new ArrayList<>();
        result.add(53648);
        result.add(59);
        byte[] bytes = text.getBytes();
        int start = 0;
        for (int i = 0; i < bytes.length; i++){
            byte[] copy = Arrays.copyOfRange(bytes, start, i + 1);
            if (!tiesSet.contains(MCUUtils.bytesToHex(copy)) || i == bytes.length - 1){
                String word = new String(Arrays.copyOfRange(bytes, start, i == bytes.length - 1 ? bytes.length : i));
                start = i;
                if (encoder.containsKey(word)) result.add(encoder.get(word));
            }
        }
        result.add(261);
        result.add(40301);
        result.add(59);
        Log.e("Dong", "encode: "+Arrays.toString(result.toArray()));
        return result;
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int i : tokens){
            if (decoder.containsKey(i)) sb.append(decoder.get(i));
        }
        return sb.toString();
    }

    private void addTies(String word){
        byte[] bytes = word.getBytes();
        for (int i = 1; i <= bytes.length; i++) {
            tiesSet.add(MCUUtils.bytesToHex(Arrays.copyOf(bytes, i)));
        }
    }

    private void fillEncoder(){
        try {
            for (Map.Entry<Integer, String> entry : decoder.entrySet()){
                encoder.put(entry.getValue(), entry.getKey());
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    private void fillDecoder(){
        try {
            String path = PathManager.getModelPath(context) + "/" + VOCAB_NAME;
            Map<String, String> map = new HashMap<>();
            map.putAll(new Gson().fromJson(new FileReader(path), decoder.getClass()));
            for (Map.Entry<String, String> entry : map.entrySet()){
                addTies(entry.getValue());
                decoder.put(Integer.parseInt(entry.getKey()), entry.getValue());
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
