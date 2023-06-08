package com.litesnap.open.rwkv;

/**
 * Created by ZTMIDGO 2022/9/9
 */
public class GPTStrategy {
    public GPTStrategyEnum strategy;
    public int value = 0;

    public GPTStrategy(GPTStrategyEnum strategy, int value) {
        this.strategy = strategy;
        this.value = value;
    }
}
