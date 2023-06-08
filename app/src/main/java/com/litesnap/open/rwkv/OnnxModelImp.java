package com.litesnap.open.rwkv;

import android.content.Context;
import android.util.Log;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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
import ai.onnxruntime.OrtSession;

/**
 * Created by ZTMIDGO 2022/9/15
 */
public class OnnxModelImp implements GptModel {
    private final String MODEL_NAME = "model.ort";
    private final OrtEnvironment environment = OrtEnvironment.getEnvironment();
    private final OrtSession.SessionOptions options = new OrtSession.SessionOptions();

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
    private final int sequenceLength = 1024;

    public OnnxModelImp(Context context, GptTokenizer tokenizer){
        this.context = context;
        this.tokenizer = tokenizer;
        try {
            String path = PathManager.getModelPath(context) + "/" + MODEL_NAME;
            options.addConfigEntry("session.load_model_format", "ORT");
            session = environment.createSession(path, options);
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    public void generate(String text, int maxCount, Callback callback) {
        if (runnable != null) runnable.setCancel(true);
        if (ort != null){
            ort.close();
            ort = null;
        }
        runnable = new MyRunnable() {
            @Override
            public void run() {
                Map<String, OnnxTensor> map = new LinkedHashMap<>();
                try {
                    float[] pp = new float[layer * embd];
                    Arrays.fill(pp, (float) -1e30);
                    OnnxTensor xx_att = OnnxTensor.createTensor(environment, FloatBuffer.wrap(new float[layer * embd]), new long[]{layer, embd});
                    OnnxTensor aa_att = OnnxTensor.createTensor(environment, FloatBuffer.wrap(new float[layer * embd]), new long[]{layer, embd});
                    OnnxTensor bb_att = OnnxTensor.createTensor(environment, FloatBuffer.wrap(new float[layer * embd]), new long[]{layer, embd});
                    OnnxTensor pp_att = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pp), new long[]{layer, embd});
                    OnnxTensor xx_ffn = OnnxTensor.createTensor(environment, FloatBuffer.wrap(new float[layer * embd]), new long[]{layer, embd});
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
                        map.put("idx", idx);
                        map.put("xx_att", xx_att);
                        map.put("aa_att", aa_att);
                        map.put("bb_att", bb_att);
                        map.put("pp_att", pp_att);
                        map.put("xx_ffn", xx_ffn);
                        Set<String> outs = new LinkedHashSet<>();
                        outs.add("x");
                        outs.add("xx_att_r");
                        outs.add("aa_att_r");
                        outs.add("bb_att_r");
                        outs.add("pp_att_r");
                        outs.add("xx_ffn_r");
                        ort = session.run(map);

                        OnnxValue run = ort.get(0);
                        xx_att = (OnnxTensor) ort.get(1);
                        aa_att = (OnnxTensor) ort.get(2);
                        bb_att = (OnnxTensor) ort.get(3);
                        pp_att = (OnnxTensor) ort.get(4);
                        xx_ffn = (OnnxTensor) ort.get(5);

                        float[] predictions = (float[]) run.getValue();
                        //ort.close();

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
                        }
                    }
                    if (ort != null){
                        ort.close();
                        ort = null;
                    }
                }catch (Exception e){
                    e.printStackTrace();
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
