package com.litesnap.open.rwkv;

import android.content.Context;

import com.google.gson.Gson;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
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
public class GptTokenizerImp implements GptTokenizer {
    private final Pattern pattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
    private final String VOCAB_NAME = "vocab.json";

    private final Map<String, Integer> encoder = new HashMap<>();
    private final Map<Integer, String> decoder = new HashMap<>();
    private final Map<Pair<String, String>, Integer> bpeRanks = new HashMap<>();
    private final Context context;

    public GptTokenizerImp(Context context){
        this.context = context;
        fillEncoder();
        fillDecoder();
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int value : tokens){
            if (decoder.containsKey(value)) sb.append(decoder.get(value));
        }
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < sb.length(); i++){
            String key = String.valueOf(sb.charAt(i));
            if (GPTByteUtils.BYTE_DECODER.containsKey(key)){
                result.add(GPTByteUtils.BYTE_DECODER.get(key));
            }
        }
        int[] ints = new int[result.size()];
        for (int i = 0; i < result.size(); i++) ints[i] = result.get(i);
        return new String(ints, 0, ints.length);
    }

    @Override
    public List<Integer> encode(String text){
        List<String> stringList = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()){
            MatchResult result = matcher.toMatchResult();
            String value = result.group();
            Stream<Integer> stream = value.codePoints().boxed();
            StringBuilder sb = new StringBuilder();
            Object[] array = stream.toArray();
            for (Object o : array){
                if (GPTByteUtils.BYTE_ENCODER.containsKey(o)){
                    sb.append(GPTByteUtils.BYTE_ENCODER.get(o));
                }
            }
            stringList.add(sb.toString());
        }

        List<List<String>> strings = new ArrayList<>();
        for (String string : stringList){
            strings.add(bpe(string));
        }

        List<Integer> result = new ArrayList<>();
        for (List<String> list : strings){
            for (String string : list) {
                if (encoder.containsKey(string)){
                    result.add(encoder.get(string));
                }
            }
        }
        return result;
    }

    private List<String> bpe(String token){
        if (token.length() <= 1) return Arrays.asList(token);

        List<String> word = new ArrayList<>(token.length());
        for (int i = 0; i < token.length(); i++) word.add(String.valueOf(token.charAt(i)));
        Set<Pair<String, String>> pairs = getPairs(word);

        while (true){
            Pair<String, String> min = null;
            int minValue = 0;
            for (Pair pair : pairs){
                if (!bpeRanks.containsKey(pair)) {
                    continue ;
                }
                int value = bpeRanks.get(pair);
                if (min == null || value < minValue){
                    min = pair;
                    minValue = value;
                }
            }

            if (min == null) break;

            int i = 0;
            List<String> newWord = new ArrayList<>();
            while (i < word.size()){
                int j = -1;
                for (int x =0; x < word.size(); x++){
                    if (x >= i && word.get(x).equals(min.first)){
                        j = x;
                        break;
                    }
                }
                if (j != -1){
                    newWord.addAll(word.subList(i, j));
                    i = j;
                }else {
                    newWord.addAll(word.subList(i, word.size()));
                    break;
                }

                if (word.get(i).equals(min.first) && i < word.size() - 1 && word.get(i + 1).equals(min.second)){
                    newWord.add(min.first + min.second);
                    i += 2;
                }else {
                    newWord.add(word.get(i));
                    i += 1;
                }
            }

            word = newWord;
            if (word.size() == 1) {
                break;
            } else {
                pairs = getPairs(word);
            }
        }
        return word;
    }

    private Set<Pair<String, String>> getPairs(List<String> word){
        Set<Pair<String, String>> result = new LinkedHashSet<>();
        for (int i =0; i < word.size() - 1; i++){
            result.add(new Pair<>(word.get(i), word.get(i + 1)));
        }
        return result;
    }

    private void fillEncoder(){
        try {
            String path = PathManager.getModelPath(context) + "/" + VOCAB_NAME;
            Gson gson = new Gson();
            Vocab vocab = gson.fromJson(new FileReader(path), Vocab.class);
            encoder.putAll(vocab.getModel().getVocab());
            fillBpeRanks(vocab.getModel().getMerges());
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    private void fillBpeRanks(String[] array){
        for (int i = 0; i < array.length; i++){
            String[] data = array[i].split(" ");
            if (data.length >= 2) {
                bpeRanks.put(new Pair<>(data[0], data[1]), i);
            }
        }
    }

    private void fillDecoder(){
        for (Map.Entry<String, Integer> entry : encoder.entrySet())
            decoder.put(entry.getValue(), entry.getKey());
    }
}
