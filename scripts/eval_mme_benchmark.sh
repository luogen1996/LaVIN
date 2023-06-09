torchrun --nproc_per_node 1 --master_port 11345 eval_mme.py \
    --ckpt_dir ../data/weights/ \
    --llm_model 7B\
    --tokenizer_path ../data/weights/tokenizer.model \
    --data_root ../data \
    --caption_file ../data/captions.json \
    --adapter_path ./15-eph-pretrain.pth \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 16\
    --max_seq_len 512 \
    --split test \
    --n_prompt 6 \
    --temperature 5.\
    --visual_adapter_type router