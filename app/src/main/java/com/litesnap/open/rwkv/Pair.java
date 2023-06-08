package com.litesnap.open.rwkv;

import java.util.Objects;

/**
 * Created by ZTMIDGO 2022/9/9
 */
public class Pair<A, B> {
    public A first;
    public B second;

    public Pair(A first, B second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Pair<A, B> pair = (Pair<A, B>) o;
        return pair.first.equals(first) && pair.second.equals(second);
    }

    @Override
    public int hashCode() {
        return Objects.hash(first, second);
    }

    @Override
    public String toString() {
        return String.format("("+ first+", "+ second+")");
    }
}
