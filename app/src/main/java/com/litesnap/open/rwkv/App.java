package com.litesnap.open.rwkv;

import android.app.Application;

/**
 * Created by ZTMIDGO 2023/6/20
 */
public class App extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        PreferencesUtils.init(this);
    }
}
