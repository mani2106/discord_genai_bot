# set HF_TOKEN=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
# $env:HF_TOKEN
# --model google/gemma-3-4b-it-qat-q4_0-gguf
docker run --runtime nvidia --gpus all --name gemma_container -v F:\Studies\hface_cache:/models `
    --env "HUGGING_FACE_HUB_TOKEN=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" `
    --env "HF_TOKEN=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" `
    -p 8000:8000 --ipc=host `
    vllm/vllm-openai:latest `
    --model llava-hf/llava-onevision-qwen2-0.5b-ov-hf `
    --gpu-memory-utilization 0.85 `
    --max_model_len 10240
# --model /models/gemma-3-4b-it-q4_0.gguf `
# --dtype float32 `

# Load and run the model:
# docker exec -it gemma_container bash -c "vllm serve google/gemma-3-4b-it-qat-q4_0-gguf"

# Call the server using curl:
# curl -X POST "http://localhost:8000/v1/chat/completions" \
# 	-H "Content-Type: application/json" \
# 	--data '{
# 		"model": "Qwen/Qwen3-0.6B",
# 		"messages": [
# 			{
# 				"role": "user",
# 				"content": [
# 					{
# 						"type": "text",
# 						"text": "Describe this image in one sentence."
# 					},
# 					{
# 						"type": "image_url",
# 						"image_url": {
# 							"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
# 						}
# 					}
# 				]
# 			}
# 		]
# 	}'