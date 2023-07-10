torchrun --nproc_per_node 8 --master_port 12345 --nproc_per_node 8 train.py \
    --llm_model 13B\
    --llama_model_path ../data/weights/ \
    --data_path ../data/alpaca_data.json \
    --max_seq_len 512 \
    --batch_size 4 \
    --accum_iter 1 \
    --epochs 15 \
    --warmup_epochs 0.2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./LaVIN-Vicuna-13B-VLIT/\
    --adapter_type router\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 5.\
    --visual_adapter_type router \
    --do_pretrain \
    --use_vicuna