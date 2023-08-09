torchrun --nproc_per_node 8 --master_port 11111 train.py \
    --llm_model 13B\
    --llama_model_path ../data/weights/ \
    --data_path ../data/alpaca_data.json \
    --max_seq_len 384 \
    --batch_size 4 \
    --accum_iter 4 \
    --epochs 4 \
    --warmup_epochs 0.1 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./LaVIN-7B-cap/\
    --adapter_type attn\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 5.\
    --visual_adapter_type router \
    --load_adapter ./15-eph-pretrain.pth \
    --do_pretrain

torchrun --nproc_per_node 1  eval_captioning.py \
    --ckpt_dir ../data/weights/ \
    --llm_model 13B\
    --tokenizer_path ../data/weights/tokenizer.model \
    --data_root ../data \
    --caption_file ../data/captions.json \
    --adapter_path ./LaVIN-7B-cap/checkpoint-3.pth \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 100\
    --max_seq_len 384 \
    --split test \
    --n_prompt 6 \
    --temperature 5.\
    --visual_adapter_type router
python summary_captioning.py