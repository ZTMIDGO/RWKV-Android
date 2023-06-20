package com.litesnap.open.rwkv;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

/**
 * Created by ZTMIDGO 2023/6/20
 */
public class MyAdapter extends RecyclerView.Adapter<MyAdapter.Holder> {
    private Context context;
    private LayoutInflater inflater;
    private List<Talk> dataList;

    public MyAdapter(Context context, LayoutInflater inflater, List<Talk> dataList) {
        this.context = context;
        this.inflater = inflater;
        this.dataList = dataList;
    }

    @NonNull
    @Override
    public Holder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        switch (viewType){
            case Talk.TYPE_QUESTION:
                return new Holder(inflater.inflate(R.layout.holder_talk_question, parent, false));
            case Talk.TYPE_ANSWER:
                return new Holder(inflater.inflate(R.layout.holder_talk_answer, parent, false));
            default:
                return null;
        }
    }

    @Override
    public void onAttachedToRecyclerView(@NonNull RecyclerView recyclerView) {
        super.onAttachedToRecyclerView(recyclerView);
        recyclerView.setItemAnimator(null);
    }

    @Override
    public void onBindViewHolder(@NonNull Holder holder, int position) {
        holder.bind(dataList.get(position));
    }

    @Override
    public int getItemViewType(int position) {
        return dataList.get(position).getType();
    }

    @Override
    public int getItemCount() {
        return dataList.size();
    }

    public void add(Talk bean){
        dataList.add(bean);
        notifyItemInserted(dataList.size());
    }

    public void clean(){
        dataList.clear();
        notifyDataSetChanged();
    }

    public class Holder extends RecyclerView.ViewHolder {
        private TextView mTextView;
        public Holder(@NonNull View itemView) {
            super(itemView);
            mTextView = itemView.findViewById(R.id.text);
        }

        public void bind(Talk bean){
            mTextView.setText(bean.getText());
        }
    }
}
