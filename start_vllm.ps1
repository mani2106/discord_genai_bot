# set HF_TOKEN=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
# $env:HF_TOKEN
# --model google/gemma-3-4b-it-qat-q4_0-gguf
docker run --runtime nvidia --gpus all --name gemma_container -v F:\Studies\hface_cache:/models `
    -p 8000:8000 --ipc=host `
    vllm/vllm-openai:latest `
    --model /models/qwen3vl_2b `
    --gpu-memory-utilization 0.85 `
    --quantization fp8 `
    --kv-cache-dtype fp8 `
    --max_model_len 3312 `
    --max-num-batched-tokens 2048
