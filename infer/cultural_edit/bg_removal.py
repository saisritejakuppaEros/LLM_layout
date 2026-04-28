from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize

model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    denom = ma - mi
    if denom.item() == 0:
        result = torch.zeros_like(result)
    else:
        result = (result - mi) / denom
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print('Removing background from image...')

# prepare input
script_dir = Path(__file__).resolve().parent
image_path = script_dir / "photos" / "srk_2.png"
out_dir = script_dir / "bg_removed"
out_dir.mkdir(parents=True, exist_ok=True)
# PNG preserves alpha; same basename as the source image
out_path = out_dir / f"{image_path.stem}.png"

orig_image = Image.open(image_path).convert("RGB")
orig_im = np.array(orig_image)
# F.interpolate expects spatial size (H, W), same as numpy image shape[:2]
orig_im_size = list(orig_im.shape[0:2])
model_input_size = [1024, 1024]
image = preprocess_image(orig_im, model_input_size).to(device)

# inference
with torch.no_grad():
    result = model(image)

# post process (mask matches original H×W)
result_image = postprocess_image(result[0][0], orig_im_size)

# save result
pil_mask_im = Image.fromarray(result_image).convert("L")
no_bg_image = orig_image.copy()
no_bg_image.putalpha(pil_mask_im)
no_bg_image.save(out_path)
print(f"Saved {out_path} ({no_bg_image.size[0]}×{no_bg_image.size[1]})")
