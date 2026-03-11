from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# use only gpu 6
import torch
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(device)

# Cache directory - models will be downloaded and cached here
cache_dir = "./models/cache"

# TEXT MODEL
text_model_name = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(text_model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(text_model_name, cache_dir=cache_dir)


# VISION MODEL
vl_model_name = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(vl_model_name, cache_dir=cache_dir)
model = Qwen3VLForConditionalGeneration.from_pretrained(vl_model_name, cache_dir=cache_dir)

print("Models downloaded and cached locally.")
