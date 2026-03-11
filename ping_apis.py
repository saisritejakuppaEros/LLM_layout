#!/usr/bin/env python3
"""Sample script to ping the text and vision APIs."""

import requests

TEXT_API = "http://localhost:8001/generate"
VL_API = "http://localhost:8002/generate"
LOCAL_IMAGE = "/mnt/data0/teja/research_multiref/InstanceAssemble/fig/teaser.jpg"


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


if __name__ == "__main__":
    print("\nPinging Qwen APIs...\n")
    t = ping_text_api()
    v = ping_vl_api()
    print("=" * 50)
    print(f"Text API: {'OK' if t else 'FAILED'}")
    print(f"Vision API: {'OK' if v else 'FAILED'}")
    print("=" * 50)
