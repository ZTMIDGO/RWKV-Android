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
    private View mInitView;
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
        mInitView = findViewById(R.id.init);
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
                            e.printStackTrace();
                        }finally {
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    dialog.dismiss();
                                }
                            });
                        }
                    }
                });
            }
        });

        mInitView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                GptTokenizer tokenizer = new GptTokenizerImp(MainActivity.this);
                model = new OnnxModelImp(MainActivity.this, tokenizer);
            }
        });

        mStartView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String text = mEditText.getText().toString();
                mTextView.setText(text);
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
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        exec.shutdownNow();
        model.close();
    }
}