package com.litesnap.open.rwkv;

import android.content.Context;
import android.util.Log;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Created by ZTMIDGO 2022/9/15
 */
public class OnnxModelImp implements GptModel {
    private final String MODEL_NAME = "model.onnx";
    private final OrtEnvironment environment = OrtEnvironment.getEnvironment();
    private final OrtSession.SessionOptions options = new OrtSession.SessionOptions();
    private final Map<String, OnnxTensor> map = new LinkedHashMap<>();
    private final Random random = new Random();
    private final GPTStrategy strategy = new GPTStrategy(GPTStrategyEnum.TOPK, 5);
    private final ExecutorService exec = Executors.newCachedThreadPool();

    private final Context context;
    private final GptTokenizer tokenizer;

    private OrtSession.Result ort;
    private OrtSession session;
    private MyRunnable runnable;

    private final int layer = 24;
    private final int embd = 1024;
    private final int sequenceLength = 1;
    private final List<String> inputNames = new ArrayList<>();

    public OnnxModelImp(Context context, GptTokenizer tokenizer){
        this.context = context;
        this.tokenizer = tokenizer;
        try {
            String path = PathManager.getModelPath(context) + "/" + MODEL_NAME;
            session = environment.createSession(path, options);
            inputNames.addAll(session.getInputNames());
            fillMap();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    public void generate(String text, int maxCount, Callback callback) {
        if (runnable != null) runnable.setCancel(true);
        //closeResult();

        runnable = new MyRunnable() {
            @Override
            public void run() {
                try {
                    List<Integer> arrays = tokenizer.encode(text);
                    List<Integer> tokens = new ArrayList<>();

                    for (int i = 0; i < maxCount; i++) {
                        int[] paddedTokens = new int[sequenceLength];
                        IntBuffer buffer = IntBuffer.wrap(paddedTokens);

                        if (!arrays.isEmpty()){
                            tokens.add(arrays.remove(0));
                        }

                        int start = paddedTokens.length - tokens.size();
                        int index = start < 0 ? tokens.size() - paddedTokens.length : 0;
                        start = start < 0 ? 0 : start;

                        for (int x = start; x < paddedTokens.length; x++){
                            paddedTokens[x] = tokens.get(index ++);
                        }

                        OnnxTensor idx = OnnxTensor.createTensor(environment, buffer, new long[]{sequenceLength});

                        map.put(inputNames.get(0), idx);

                        ort = session.run(map);

                        float[] predictions = (float[]) ort.get(0).getValue();
                        for (int x = 0; x < inputNames.size(); x++){
                            map.put(inputNames.get(x),(OnnxTensor) ort.get(x));
                        }

                        if (!arrays.isEmpty()) continue;

                        if (isCancel()) return;

                        float[] outputLogits = predictions;
                        int nextToken = -1;

                        switch (strategy.strategy) {
                            case TOPK:
                                List<Pair<Integer, Float>> filteredLogitsWithIndexes = new ArrayList<>();
                                for (int x = 0; x < outputLogits.length; x++) {
                                    filteredLogitsWithIndexes.add(new Pair<>(x, outputLogits[x]));
                                }

                                Collections.sort(filteredLogitsWithIndexes, new Comparator<Pair<Integer, Float>>() {
                                    @Override
                                    public int compare(Pair<Integer, Float> o1, Pair<Integer, Float> o2) {
                                        if (o1.second > o2.second) {
                                            return -1;
                                        } else if (o1.second < o2.second) {
                                            return 1;
                                        } else {
                                            return 0;
                                        }
                                    }
                                });

                                if (filteredLogitsWithIndexes.size() > strategy.value)
                                    filteredLogitsWithIndexes = filteredLogitsWithIndexes.subList(0, strategy.value);

                                List<Float> filteredLogits = new ArrayList<>(filteredLogitsWithIndexes.size());
                                for (Pair<Integer, Float> pair : filteredLogitsWithIndexes)
                                    filteredLogits.add(pair.second);

                                float maxLogitValue = filteredLogits.get(0);
                                float sumExp = 0;

                                List<Float> logitsExp = new ArrayList<>();
                                for (float value : filteredLogits) {
                                    float result = (float) Math.exp(value - maxLogitValue);
                                    sumExp += result;
                                    logitsExp.add(result);
                                }

                                List<Float> probs = new ArrayList<>();
                                for (float value : logitsExp) probs.add(value / sumExp);

                                List<Integer> logitsIndexes = new ArrayList<>();
                                for (Pair<Integer, Float> pair : filteredLogitsWithIndexes)
                                    logitsIndexes.add(pair.first);

                                nextToken = sample(logitsIndexes, probs);
                                break;
                            default:
                                float value = 0;
                                for (int x = 0; x < outputLogits.length; x++) {
                                    float data = outputLogits[x];
                                    if (x == 0) {
                                        value = data;
                                        nextToken = 0;
                                    } else if (data > value) {
                                        value = data;
                                        nextToken = x;
                                    }
                                }
                        }

                        if (arrays.isEmpty()){
                            tokens.add(nextToken);
                            String decodedToken = tokenizer.decode(Arrays.asList(nextToken));
                            if (callback != null) callback.callback(decodedToken);
                            Log.e("Dong", "run: "+nextToken+"  "+decodedToken);
                            if (nextToken == 261 || nextToken == 0) break;
                        }
                    }
                }catch (Exception e){
                    e.printStackTrace();
                }finally {
                    //closeResult();
                }
            }
        };
        exec.execute(runnable);
    }

    @Override
    public int sample(List<Integer> indexes, List<Float> probs){
        int index = randomIndex(probs);
        return indexes.get(index);
    }

    @Override
    public void close() {
        if (runnable != null) runnable.setCancel(true);
        exec.shutdown();
        if (session != null) {
            try {
                closeResult();
                session.close();
                options.close();
            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }

    @Override
    public void cancel() {
        if (runnable != null) runnable.setCancel(true);
    }

    @Override
    public void setTopK(int value) {
        strategy.value = value;
    }

    private void fillMap() throws Exception {
        for (String name : inputNames){
            float[] buff = new float[layer * embd];
            if (name.equals("pp_att")) Arrays.fill(buff, (float) -1e30);
            OnnxTensor inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(buff), new long[]{layer, embd});
            map.put(name, inputTensor);
        }
    }

    private void closeResult(){
        if (ort != null){
            ort.close();
            ort = null;
        }
    }

    private int randomIndex(List<Float> probs){
        float sun = 0;
        for (float value : probs) sun += value;
        float rnd = sun * random.nextFloat();
        float acc = 0f;
        for (int i = 0; i < probs.size(); i++){
            acc += probs.get(i);
            if (rnd < acc) return i;
        }
        return probs.size() - 1;
    }
}
