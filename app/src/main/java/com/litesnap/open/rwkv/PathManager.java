package com.litesnap.open.rwkv;

import android.content.Context;

/**
 * Created by ZTMIDGO 2023/6/8
 */
public class PathManager {
    public static String getModelPath(Context context){
        return context.getFilesDir().getAbsolutePath() + "/model";
    }
}
