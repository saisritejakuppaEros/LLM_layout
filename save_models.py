from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# use only gpu 6
import torch
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(device)

# Cache directory - models will be downloaded and cached here
cache_dir = "./models/cache"

# # TEXT MODEL
# text_model_name = "Qwen/Qwen3-8B"

# tokenizer = AutoTokenizer.from_pretrained(text_model_name, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(text_model_name, cache_dir=cache_dir)


# # VISION MODEL
# vl_model_name = "Qwen/Qwen3-VL-8B-Instruct"

# processor = AutoProcessor.from_pretrained(vl_model_name, cache_dir=cache_dir)
# model = Qwen3VLForConditionalGeneration.from_pretrained(vl_model_name, cache_dir=cache_dir)

# print("Models downloaded and cached locally.")

# # IMAGE MODEL (DiffusionPipeline)
# from diffusers import DiffusionPipeline

# image_model_name = "Qwen/Qwen-Image"
# pipe = DiffusionPipeline.from_pretrained(
#     image_model_name,
#     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#     cache_dir=cache_dir,
# )
# # Save to local directory for offline use
# save_dir = "./models/Qwen-Image"
# pipe.save_pretrained(save_dir)
# print(f"Image model saved to {save_dir}")

# IMAGE EDIT MODEL (QwenImageEditPlusPipeline)
from diffusers import QwenImageEditPlusPipeline

image_edit_model_name = "Qwen/Qwen-Image-Edit-2511"
edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
    image_edit_model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    cache_dir=cache_dir,
)
edit_save_dir = "./models/Qwen-Image-Edit-2511"
edit_pipe.save_pretrained(edit_save_dir)
print(f"Image edit model saved to {edit_save_dir}")