package com.litesnap.open.rwkv;

import android.app.ProgressDialog;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {
    private final ExecutorService exec = Executors.newCachedThreadPool();

    private OnnxModelImp model;
    private TextView mTextView;
    private View mStartView;

    private EditText mNumView;
    private EditText mTopKView;
    private EditText mEditText;
    private View mCreateView;

    private ProgressDialog dialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mTextView = findViewById(R.id.text);
        mEditText = findViewById(R.id.edit);
        mStartView = findViewById(R.id.start);
        mNumView = findViewById(R.id.number);
        mTopKView = findViewById(R.id.topk);
        mCreateView = findViewById(R.id.create);
        dialog = new ProgressDialog(this);

        mCreateView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dialog.show();
                exec.execute(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            FileUtils.copyAssets(getAssets(), "model", getFilesDir().getAbsoluteFile());
                        }catch (Exception e){
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    mTextView.setText("复制模型失败");
                                }
                            });
                        }finally {
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    mTextView.setText("已完成模型复制");
                                    dialog.dismiss();
                                }
                            });
                        }
                    }
                });
            }
        });

        mStartView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (model != null && model.isRunning()) return;

                if (model == null){
                    mTextView.setText("初始化中. . .");
                    exec.execute(new Runnable() {
                        @Override
                        public void run() {
                            GptTokenizer tokenizer = new WorldTokenizerImp(MainActivity.this);
                            model = new OnnxModelImp(MainActivity.this, tokenizer);
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    mTextView.setText("初始化完成");
                                    working();
                                }
                            });
                        }
                    });
                }else {
                    working();
                }
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        exec.shutdownNow();
        model.close();
    }

    private void working(){
        String text = mEditText.getText().toString();
        String line = String.format("Question:%s\n\nAnswer:", text);
        mTextView.setText(line);
        final int number = Integer.parseInt(mNumView.getText().toString());
        final int topk = Integer.parseInt(mTopKView.getText().toString());
        model.setTopK(topk);
        model.generate(text, number, new GptModel.Callback() {
            @Override
            public void callback(String text) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        mTextView.append(text);
                    }
                });
            }
        });
    }
}