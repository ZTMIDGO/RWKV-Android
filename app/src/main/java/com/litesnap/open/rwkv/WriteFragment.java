package com.litesnap.open.rwkv;

import android.app.ProgressDialog;
import android.os.Bundle;
import android.os.Handler;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ScrollView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by ZTMIDGO 2023/6/20
 */
public class WriteFragment extends Fragment {
    private final ExecutorService exec = Executors.newCachedThreadPool();

    public static WriteFragment newInstance() {

        Bundle args = new Bundle();

        WriteFragment fragment = new WriteFragment();
        fragment.setArguments(args);
        return fragment;
    }

    private EditText mContentView;
    private EditText mTopKView;
    private EditText mLenView;
    private EditText mP1View;
    private EditText mP2View;
    private EditText mTempView;
    private EditText mTopPView;
    private View mStartView;
    private ScrollView mScrollView;

    private Handler uiHandler;
    private ProgressDialog dialog;
    private GptTokenizer tokenizer;
    private OnnxModelImp model;

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        uiHandler = new Handler();
        dialog = new ProgressDialog(getActivity());
        dialog.setCancelable(false);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        exec.shutdownNow();
        if (model != null) model.close();
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_write, container, false);
        mContentView = view.findViewById(R.id.content);
        mTopKView = view.findViewById(R.id.top_k);
        mLenView = view.findViewById(R.id.len);
        mP1View = view.findViewById(R.id.p1);
        mP2View = view.findViewById(R.id.p2);
        mStartView = view.findViewById(R.id.start);
        mScrollView = view.findViewById(R.id.scroll);
        mTempView = view.findViewById(R.id.temp);
        mTopPView = view.findViewById(R.id.top_p);

        mTopKView.setText(String.valueOf(PreferencesManager.getTopK()));
        mLenView.setText(String.valueOf(PreferencesManager.getLen()));
        mP1View.setText(String.valueOf(PreferencesManager.getP1()));
        mP2View.setText(String.valueOf(PreferencesManager.getP2()));
        mTempView.setText(String.valueOf(PreferencesManager.getTemp()));
        mTopPView.setText(String.valueOf(PreferencesManager.getTopp()));

        mStartView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String topKStr = mTopKView.getText().toString();
                String lenStr = mLenView.getText().toString();
                String p1Str = mP1View.getText().toString();
                String p2Str = mP2View.getText().toString();
                String text = mContentView.getText().toString();
                String tempStr = mTempView.getText().toString();
                String toppStr = mTopPView.getText().toString();

                final int topK = TextUtils.isEmpty(topKStr) ? PreferencesManager.getTopK() : Integer.parseInt(topKStr);
                final int len = TextUtils.isEmpty(lenStr) ? PreferencesManager.getLen() : Integer.parseInt(lenStr);
                final float p1 = TextUtils.isEmpty(p1Str) ? PreferencesManager.getP1() : Float.parseFloat(p1Str);
                final float p2 = TextUtils.isEmpty(p2Str) ? PreferencesManager.getP2() : Float.parseFloat(p2Str);
                final float temp = TextUtils.isEmpty(tempStr) ? PreferencesManager.getTemp() : Float.parseFloat(tempStr);
                final float topp = TextUtils.isEmpty(toppStr) ? PreferencesManager.getTopp() : Float.parseFloat(toppStr);


                PreferencesUtils.setProperty(Atts.LEN, (int)len);
                PreferencesUtils.setProperty(Atts.TOP_K, (int)topK);
                PreferencesUtils.setProperty(Atts.P1, p1 * 1f);
                PreferencesUtils.setProperty(Atts.P2, p2 * 1f);
                PreferencesUtils.setProperty(Atts.TEMP, temp * 1f);
                PreferencesUtils.setProperty(Atts.TOP_P, topp * 1f);

                if (model != null && model.isRunning()) return;

                if (model == null){
                    dialog.show();
                    exec.execute(new MyRunnable() {
                        @Override
                        public void run() {
                            tokenizer = new WorldTokenizerImp(getActivity());
                            model = new OnnxModelImp(getActivity(), OnnxModelImp.MODE_WRITE);
                            uiHandler.post(new Runnable() {
                                @Override
                                public void run() {
                                    working(text, temp, topp, topK, len, p1, p2);
                                }
                            });
                        }
                    });
                }else {
                    working(text, temp, topp, topK, len, p1, p2);
                }
            }
        });

        return view;
    }

    private void working(String text, float temp, float topp, int topK, int len, float p1, float p2){
        dialog.dismiss();
        model.setTop(temp, topp, topK);
        model.setPenalty(p1, p2);
        List<Integer> integers = new ArrayList<>();
        integers.add(11);
        integers.addAll(tokenizer.encode(text));
        model.generate(integers, len, new GptModel.Callback() {
            @Override
            public void callback(int token, int index, int maxCount, boolean isEnd) {
                uiHandler.post(new MyRunnable() {
                    @Override
                    public void run() {
                        mContentView.append(tokenizer.decode(Arrays.asList(token)));
                        mScrollView.fullScroll(ScrollView.FOCUS_DOWN);
                    }
                });
            }
        });
    }
}
