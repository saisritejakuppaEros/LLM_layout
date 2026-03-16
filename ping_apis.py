#!/usr/bin/env python3
"""Sample script to ping the text and vision APIs."""

import base64
import requests

TEXT_API = "http://localhost:8001/generate"
VL_API = "http://localhost:8002/generate"
IMAGE_API = "http://localhost:8004/generate"
LOCAL_IMAGE = "/mnt/data0/harsha/synthetic_dataset_gen/bucket_samples/02_clothing_textiles/02_clothing_textiles_08.jpg"
IMAGE_EDIT_TEST_IMAGE = "/mnt/data0/harsha/synthetic_dataset_gen/bucket_samples/02_clothing_textiles/02_clothing_textiles_04.jpg"


def ping_text_api():
    """Ping the text model API."""
    print("=" * 50)
    print("Pinging Text API (port 8001)...")
    print("=" * 50)
    try:
        resp = requests.post(
            TEXT_API,
            json={"prompt": "Explain transformers in one sentence.", "max_tokens": 64},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        print("Response:", data.get("response", data)[:200] + "..." if len(str(data.get("response", data))) > 200 else data.get("response", data))
        print("OK\n")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect. Is the text API running on port 8001?\n")
        return False
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False


def ping_vl_api():
    """Ping the vision model API with local image."""
    print("=" * 50)
    print("Pinging Vision API (port 8002)...")
    print("=" * 50)
    try:
        resp = requests.post(
            VL_API,
            json={
                "image_url": LOCAL_IMAGE,
                "prompt": "Describe this image briefly.",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        response = data.get("response", data)
        print("Response:", response[:300] + "..." if len(str(response)) > 300 else response)
        print("OK\n")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect. Is the vision API running on port 8002?\n")
        return False
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False


def ping_image_api():
    """Ping the image edit API (image path + prompt)."""
    print("=" * 50)
    print("Pinging Image Edit API (port 8004)...")
    print("=" * 50)
    try:
        resp = requests.post(
            IMAGE_API,
            json={
                "prompt": "a realistic image of a man wearing a red nehru jacket, preserve the fabric texture",
                "image_path": IMAGE_EDIT_TEST_IMAGE,
                "negative_prompt": "blurry, low quality, distorted, texture change, fabric distortion",
                "num_inference_steps": 50,
                "width": 1536,
                "height": 1024,
                "true_cfg_scale": 4.0,
                "seed": 42,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        image_b64 = data.get("response", "")
        out_path = "image_edit_output.png"
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        print(f"Response: base64 image ({len(image_b64)} chars)")
        print(f"Saved to {out_path}\n")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect. Is the image API running on port 8004?\n")
        return False
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False


if __name__ == "__main__":
    print("\nPinging Qwen APIs...\n")
    # t = ping_text_api()
    # v = ping_vl_api()
    i = ping_image_api()
    print("=" * 50)
    print("=" * 50)
