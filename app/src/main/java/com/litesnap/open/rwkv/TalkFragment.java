package com.litesnap.open.rwkv;

import android.app.ProgressDialog;
import android.os.Bundle;
import android.os.Handler;
import android.text.TextUtils;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by ZTMIDGO 2023/6/20
 */
public class TalkFragment extends Fragment {
    private final ExecutorService exec = Executors.newCachedThreadPool();

    public static TalkFragment newInstance() {

        Bundle args = new Bundle();

        TalkFragment fragment = new TalkFragment();
        fragment.setArguments(args);
        return fragment;
    }

    private EditText mTopKView;
    private EditText mLenView;
    private EditText mP1View;
    private EditText mP2View;
    private EditText mTempView;
    private EditText mTopPView;
    private View mCleanView;
    private View mSendView;
    private EditText mEditText;
    private RecyclerView mRecyclerView;
    private LinearLayoutManager mLayoutManager;

    private Handler uiHandler;
    private ProgressDialog dialog;
    private GptTokenizer tokenizer;
    private OnnxModelImp model;
    private MyAdapter mAdapter;

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
        View view = inflater.inflate(R.layout.fragment_answer, container, false);
        mTopKView = view.findViewById(R.id.top_k);
        mLenView = view.findViewById(R.id.len);
        mP1View = view.findViewById(R.id.p1);
        mP2View = view.findViewById(R.id.p2);
        mCleanView = view.findViewById(R.id.clean);
        mSendView = view.findViewById(R.id.send);
        mEditText = view.findViewById(R.id.edit);
        mRecyclerView = view.findViewById(R.id.recycler_view);
        mTempView = view.findViewById(R.id.temp);
        mTopPView = view.findViewById(R.id.top_p);

        mLayoutManager = new LinearLayoutManager(getActivity(), RecyclerView.VERTICAL, false);
        mAdapter = new MyAdapter(getActivity(), inflater, new ArrayList<>());
        mRecyclerView.setLayoutManager(mLayoutManager);
        mLayoutManager.setStackFromEnd(true);
        mRecyclerView.setAdapter(mAdapter);

        mTopKView.setText(String.valueOf(PreferencesManager.getTopK()));
        mLenView.setText(String.valueOf(PreferencesManager.getLen()));
        mP1View.setText(String.valueOf(PreferencesManager.getP1()));
        mP2View.setText(String.valueOf(PreferencesManager.getP2()));
        mTempView.setText(String.valueOf(PreferencesManager.getTemp()));
        mTopPView.setText(String.valueOf(PreferencesManager.getTopp()));

        mSendView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String topKStr = mTopKView.getText().toString();
                String lenStr = mLenView.getText().toString();
                String p1Str = mP1View.getText().toString();
                String p2Str = mP2View.getText().toString();
                String text = mEditText.getText().toString();
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
                            model = new OnnxModelImp(getActivity(), OnnxModelImp.MODE_TALK);
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

        mCleanView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (model == null || model.isRunning()) return;
                
                mAdapter.clean();
                model.clean();
            }
        });

        return view;
    }

    private void working(String text, float temp, float topp, int topK, int len, float p1, float p2){
        dialog.dismiss();

        if (TextUtils.isEmpty(text)) return;

        model.setTop(temp, topp, topK);
        model.setPenalty(p1, p2);
        mAdapter.add(new Talk(Talk.TYPE_QUESTION, text));
        final Talk answer = new Talk(Talk.TYPE_ANSWER, "");
        mAdapter.add(answer);
        mLayoutManager.scrollToPositionWithOffset(mAdapter.getItemCount() - 1, Integer.MIN_VALUE);

        List<Integer> integers = new ArrayList<>();
        integers.add(11);
        integers.add(261);
        integers.add(53648);
        integers.add(59);
        integers.addAll(tokenizer.encode(text));
        integers.add(261);
        integers.add(40301);
        integers.add(59);

        mEditText.setText("");
        model.generate(integers, len, new GptModel.Callback() {
            long time = System.currentTimeMillis();
            int laster = 0;
            @Override
            public void callback(int token, int index, int maxCount, boolean isEnd) {
                if (TimeUnit.MILLISECONDS.toMillis(System.currentTimeMillis() - time) >= 1000){
                    Log.e("Dong", "callback: 每秒生成 "+(index - laster));
                    time = System.currentTimeMillis();
                    laster = index;
                }
                answer.setText((answer.getText() + tokenizer.decode(Arrays.asList(token))).replaceFirst("(\n\n)$", ""));
                uiHandler.post(new MyRunnable() {
                    @Override
                    public void run() {
                        mAdapter.notifyItemChanged(mAdapter.getItemCount() - 1);
                        mLayoutManager.scrollToPositionWithOffset(mAdapter.getItemCount() - 1, Integer.MIN_VALUE);
                    }
                });
            }
        });
    }
}
