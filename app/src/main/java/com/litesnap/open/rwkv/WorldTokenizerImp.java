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
    private final String VOCAB_NAME = "vocab.txt";
    private final Map<String, Integer> encoder = new HashMap<>();
    private final Map<Integer, String> decoder = new HashMap<>();
    private final Context context;

    public WorldTokenizerImp(Context context){
        this.context = context;
        fillEncoder();
        fillDecoder();
    }

    @Override
    public List<Integer> encode(String text) {
        List<Integer> result = new ArrayList<>(text.length() + 1);
        result.add(53648);
        result.add(59);
        String[] array = StringUtils.toArrays(text);
        for (int i = 0; i < array.length; i++){
            String c = array[i];
            if (encoder.containsKey(c)) result.add(encoder.get(c));
        }
        result.add(261);
        result.add(40301);
        result.add(59);
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

    private void fillEncoder(){
        try {
            String path = PathManager.getModelPath(context) + "/" + VOCAB_NAME;
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line;
            Pattern pattern = Pattern.compile("^([0-9]*) ");
            while ((line = reader.readLine()) != null){
                line = line.replaceFirst(" ([0-9]*)$", "");
                Matcher matcher = pattern.matcher(line);
                if (matcher.find()){
                    String key = matcher.group();
                    line = line.replaceFirst(key, "");
                    key = key.replaceAll(" ", "");
                    if (line.startsWith("'")){
                        line = line.replaceFirst("^'", "");
                        line = line.replaceFirst("'$", "");
                    }
                    encoder.put(line, Integer.parseInt(key));
                }
            }
            reader.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    private void fillDecoder(){
        for (Map.Entry<String, Integer> entry : encoder.entrySet())
            decoder.put(entry.getValue(), entry.getKey());
    }
}
