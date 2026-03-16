# this is aobut llm based layout generation.

The goal is to use an llm to do layout generation.
Task1: ask the llm to decide based on the theme, about the objects in the foreground, midground and background.

Steps:
to get the models
1. save_models.py

2. run the models


```
conda activate qwen_parth
```

```
cd /mnt/data0/teja/research_multiref/llm_based_layout
CUDA_VISIBLE_DEVICES=4 uvicorn api_qwen_text:app --host 0.0.0.0 --port 8001
```

```
cd /mnt/data0/teja/research_multiref/llm_based_layout
CUDA_VISIBLE_DEVICES=4 uvicorn api_qwen_vl:app --host 0.0.0.0 --port 8002
```

```
cd /mnt/data0/teja/research_multiref/llm_based_layout
CUDA_VISIBLE_DEVICES=4 uvicorn api_qwen_image:app --host 0.0.0.0 --port 8004
CUDA_VISIBLE_DEVICES=4 uvicorn api_qwen_image:app --host 0.0.0.0 --port 8009
```

python -m stage2.pipeline outputs/stage1/locations_prompts.json