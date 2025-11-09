#!/bin/bash
cd /root/LLaVA-LayerRouter-ca
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa_loader \
    --model-path /hy-tmp/checkpoints/llava-v1.5-7b \
    --question-file /hy-tmp/TextVQA/test.jsonl \
    --image-folder /hy-tmp/TextVQA/train_images \
    --answers-file /hy-tmp/TextVQA/llava-v1.5-7b_answers.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file /hy-tmp/TextVQA/TextVQA_0.5.1_val.jsonl \
#     --result-file /home/data/shika/LLaVA//playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl
