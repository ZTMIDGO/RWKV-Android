package com.litesnap.open.rwkv;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

public class SampleLogits {
    public static final Random RANDOM = new Random();
    public static int sample(float[] logits, float temperature, float top_p, int top_k){
        float[] probs = softmax(logits);
        int[] sorted_ids = argsort(probs, true);
        float[] sorted_probs = sit(probs, sorted_ids);
        float[] cumulative_probs = cumsum(sorted_probs);
        float cutoff = sorted_probs[argmax(cumulative_probs, top_p)];

        for (int i = 0; i < probs.length; i++){
            if (probs[i] < cutoff) probs[i] = 0;
        }

        if (top_k < probs.length && top_k > 0){
            for (int i = 0; i < sorted_ids.length - top_k; i++){
                probs[sorted_ids[i]] = 0;
            }
        }else if (temperature != 1){
            for (int i = 0; i < probs.length; i++){
                probs[i] = (float) Math.pow(probs[i], 1.0 / temperature);
            }
        }

        float sum = sum(probs);
        for (int i = 0; i < probs.length; i++){
            probs[i] = probs[i] / sum;
        }

        int[] indexs = new int[probs.length];
        for (int i = 0; i < indexs.length; i++) indexs[i] = i;
        return choice(indexs, probs);
    }

    public static float[] softmax(float[] input) {
        float total = 0.0f;
        for (float value : input) {
            total += (float) Math.exp(value);
        }
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.exp(input[i]) / total;
        }
        return output;
    }

    public static int[] argsort(final float[] a, final boolean ascending) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Float.compare(a[i1], a[i2]);
            }
        });
        return asArray(indexes);
    }

    public static <T extends Number> int[] asArray(final T... a) {
        int[] b = new int[a.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = a[i].intValue();
        }
        return b;
    }

    public static float[] sit(float[] floats, int[] ids){
        float[] result = new float[floats.length];
        for (int i = 0; i < ids.length; i++){
            result[(ids.length - 1) - i] = floats[ids[i]];
        }
        return result;
    }

    public static float[] cumsum(float[] input) {
        float[] output = new float[input.length];
        float cumulativeSum = 0.0f;
        for (int i = 0; i < input.length; i++) {
            cumulativeSum += input[i];
            output[i] = cumulativeSum;
        }
        return output;
    }

    public static int argmax(float[] input, float value) {
        for (int i = 0; i < input.length - 1; i++) {
            if (input[i] > value) {
                return i;
            }
        }
        return 0;
    }

    public static float sum(float[] input) {
        float total = 0.0f;
        for (float value : input) {
            total += value;
        }
        return total;
    }


    public static int choice(int[] a, float[] p) {
        if (a.length != p.length) {
            throw new IllegalArgumentException("a and p must have the same length");
        }

        float r = RANDOM.nextFloat();
        float cumulativeProbability = 0.0f;
        for (int i = 0; i < a.length; i++) {
            cumulativeProbability += p[i];
            if (r <= cumulativeProbability) {
                return a[i];
            }
        }
        return -1;
    }
}
