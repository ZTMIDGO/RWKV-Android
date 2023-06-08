package com.litesnap.open.rwkv;

import java.util.Map;

/**
 * Created by ZTMIDGO 2023/6/8
 */
public class Vocab {
    private Inner model;

    public Inner getModel() {
        return model;
    }

    public class Inner{
        private Map<String, Integer> vocab;
        private String[] merges;

        public Map<String, Integer> getVocab() {
            return vocab;
        }

        public String[] getMerges() {
            return merges;
        }
    }
}
