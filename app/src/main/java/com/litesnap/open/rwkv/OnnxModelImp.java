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
    public static final int MODE_WRITE = 0;
    public static final int MODE_TALK = 1;

    private final String MODEL_NAME = "model.onnx";
    private final OrtEnvironment environment = OrtEnvironment.getEnvironment();
    private final OrtSession.SessionOptions options = new OrtSession.SessionOptions();
    private final Map<String, OnnxTensor> map = new LinkedHashMap<>();
    private final Random random = new Random();
    private final ExecutorService exec = Executors.newCachedThreadPool();

    private final Context context;

    private float temp = 1f;
    private float topp = 0.1f;
    private int topk = 0;

    private final int layer = 24;
    private final int embd = 1024;
    private final int sequenceLength = 1;
    private final List<String> inputNames = new ArrayList<>();

    private OrtSession.Result ort;
    private OrtSession session;
    private MyRunnable runnable;

    private int mode = MODE_TALK;
    private float presence = 0.7f;
    private float frequency = 0.4f;
    private boolean isRunnable;

    public OnnxModelImp(Context context, int mode){
        this.context = context;
        this.mode = mode;
        try {
            String path = PathManager.getModelPath(context) + "/" + MODEL_NAME;
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
            session = environment.createSession(path, options);
            inputNames.addAll(session.getInputNames());
            fillMap();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    public void generate(List<Integer> arrays, int maxCount, Callback callback) {
        if (runnable != null) runnable.setCancel(true);
        isRunnable = true;
        closeResult();
        runnable = new MyRunnable() {
            @Override
            public void run() {
                try {
                    Map<Integer, Float> occurrence = new HashMap<>();

                    if (mode == MODE_WRITE) fillMap();

                    int nextToken = 0;
                    int size = maxCount + arrays.size();

                    for (int i = 0; i < size; i++) {
                        int[] paddedTokens = new int[sequenceLength];
                        IntBuffer buffer = IntBuffer.wrap(paddedTokens);

                        if (!arrays.isEmpty()){
                            nextToken = arrays.remove(0);
                        }

                        paddedTokens[0] = nextToken;

                        OnnxTensor idx = OnnxTensor.createTensor(environment, buffer, new long[]{sequenceLength});

                        map.put(inputNames.get(0), idx);

                        ort = session.run(map);
                        float[] outputLogits = (float[]) ort.get(0).getValue();
                        
                        for (Map.Entry<Integer, Float> entry : occurrence.entrySet()){
                            int x = entry.getKey();
                            outputLogits[x] = outputLogits[x] - (presence + entry.getValue() * frequency);
                        }

                        nextToken = SampleLogits.sample(outputLogits, temp, topp, topk);

                        if (isCancel()) return;

                        if (!occurrence.containsKey(nextToken)) occurrence.put(nextToken, 0f);
                        occurrence.put(nextToken, occurrence.get(nextToken) + 1f);

                        if (arrays.isEmpty()){
                            if (mode == MODE_TALK && (nextToken == 60807 || nextToken == 23692 || nextToken == 33161 || nextToken == 82 || nextToken == 24281 || nextToken == 53648 || nextToken == 40301)) break;
                            if (callback != null) callback.callback(nextToken, i, maxCount, false);
                            fillMap(ort);
                        }else {
                            fillMap(ort);
                        }
                    }
                }catch (Exception e){
                    e.printStackTrace();
                }finally {
                    isRunnable = false;
                    closeResult();
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
    public void setTop(float temp, float topp, int topk) {
        this.temp = temp;
        this.topp = topp;
        this.topk = topk;
    }

    @Override
    public void setPenalty(float v1, float v2) {
        presence = v1;
        frequency = v2;
    }

    @Override
    public void clean() {
        try {
            fillMap();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public boolean isRunning() {
        return isRunnable;
    }

    private void fillMap() throws Exception {
        for (String name : inputNames){
            float[] buff = new float[layer * embd];
            if (name.equals("pp_att")) Arrays.fill(buff, (float) -1e30);
            OnnxTensor inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(buff), new long[]{layer, embd});
            map.put(name, inputTensor);
        }
    }

    /*private void fillMap() throws Exception {
        for (int i = 0; i < inputNames.size(); i++){
            String name = inputNames.get(i);
            float[] buff = new float[embd];
            OnnxTensor inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(buff), new long[]{embd});
            map.put(name, inputTensor);
        }
    }*/

    private void fillMap(OrtSession.Result result){
        if (result == null) return;
        for (int x = 0; x < inputNames.size(); x++){
            map.put(inputNames.get(x),(OnnxTensor) result.get(x));
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
