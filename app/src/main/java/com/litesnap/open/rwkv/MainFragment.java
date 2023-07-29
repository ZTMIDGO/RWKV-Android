package com.litesnap.open.rwkv;

import android.app.ProgressDialog;
import android.os.Bundle;
import android.os.Handler;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by ZTMIDGO 2023/6/20
 */
public class MainFragment extends Fragment {
    private final ExecutorService exec = Executors.newCachedThreadPool();

    public static MainFragment newInstance() {
        
        Bundle args = new Bundle();
        
        MainFragment fragment = new MainFragment();
        fragment.setArguments(args);
        return fragment;
    }

    private View mCopyView;
    private View mWriteView;
    private View mAnswerView;

    private ProgressDialog dialog;
    private Handler uiHandler;

    private boolean isCopy = false;

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
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_main, container, false);
        mCopyView = view.findViewById(R.id.copy);
        mWriteView = view.findViewById(R.id.write);
        mAnswerView = view.findViewById(R.id.answer);

        File file = new File(PathManager.getModelPath(getActivity()) + "/model.onnx");
        isCopy = file.exists();
        setEnable(isCopy);

        mCopyView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dialog.show();
                exec.execute(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            FileUtils.copyAssets(getActivity().getAssets(), "model", getActivity().getFilesDir().getAbsoluteFile());
                            isCopy = true;
                        }catch (Exception e){
                            isCopy = false;
                        }finally {
                            uiHandler.post(new Runnable() {
                                @Override
                                public void run() {
                                    setEnable(isCopy);
                                    dialog.dismiss();
                                }
                            });
                        }
                    }
                });
            }
        });

        mWriteView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getActivity().getSupportFragmentManager().beginTransaction().add(R.id.container, WriteFragment.newInstance()).addToBackStack(null).commit();
            }
        });

        mAnswerView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getActivity().getSupportFragmentManager().beginTransaction().add(R.id.container, TalkFragment.newInstance()).addToBackStack(null).commit();
            }
        });
        return view;
    }

    private void setEnable(boolean isEnable){
        mWriteView.setEnabled(isEnable);
        mAnswerView.setEnabled(isEnable);
    }
}
