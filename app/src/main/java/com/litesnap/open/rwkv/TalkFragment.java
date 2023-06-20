package com.litesnap.open.rwkv;

import android.app.ProgressDialog;
import android.os.Bundle;
import android.os.Handler;
import android.text.TextUtils;
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

        mLayoutManager = new LinearLayoutManager(getActivity(), RecyclerView.VERTICAL, false);
        mAdapter = new MyAdapter(getActivity(), inflater, new ArrayList<>());
        mRecyclerView.setLayoutManager(mLayoutManager);
        mLayoutManager.setStackFromEnd(true);
        mRecyclerView.setAdapter(mAdapter);

        mTopKView.setText(String.valueOf(PreferencesManager.getTopK()));
        mLenView.setText(String.valueOf(PreferencesManager.getLen()));
        mP1View.setText(String.valueOf(PreferencesManager.getP1()));
        mP2View.setText(String.valueOf(PreferencesManager.getP2()));

        mSendView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String topKStr = mTopKView.getText().toString();
                String lenStr = mLenView.getText().toString();
                String p1Str = mP1View.getText().toString();
                String p2Str = mP2View.getText().toString();
                String text = mEditText.getText().toString();

                final int topK = TextUtils.isEmpty(topKStr) ? PreferencesManager.getTopK() : Integer.parseInt(topKStr);
                final int len = TextUtils.isEmpty(lenStr) ? PreferencesManager.getLen() : Integer.parseInt(lenStr);
                final float p1 = TextUtils.isEmpty(p1Str) ? PreferencesManager.getP1() : Float.parseFloat(p1Str);
                final float p2 = TextUtils.isEmpty(p2Str) ? PreferencesManager.getP2() : Float.parseFloat(p2Str);

                PreferencesUtils.setProperty(Atts.TOP_K, topK);
                PreferencesUtils.setProperty(Atts.LEN, len);
                PreferencesUtils.setProperty(Atts.P1, p1);
                PreferencesUtils.setProperty(Atts.P2, p2);

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
                                    working(text, topK, len, p1, p2);
                                }
                            });
                        }
                    });
                }else {
                    working(text, topK, len, p1, p2);
                }
            }
        });

        mCleanView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (model != null && model.isRunning()) return;
                mAdapter.clean();
                model.clean();
            }
        });

        return view;
    }

    private void working(String text, int topK, int len, float p1, float p2){
        dialog.dismiss();

        if (TextUtils.isEmpty(text)) return;

        model.setTopK(topK);
        model.setPenalty(p1, p2);
        mAdapter.add(new Talk(Talk.TYPE_QUESTION, text));
        final Talk answer = new Talk(Talk.TYPE_ANSWER, "");
        mAdapter.add(answer);

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
            @Override
            public void callback(List<Integer> tokens) {
                answer.setText((answer.getText() + tokenizer.decode(tokens)).replaceFirst("(\n\n)$", ""));
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
