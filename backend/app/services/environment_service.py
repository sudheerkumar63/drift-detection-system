import io
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_environmental_effects(image_bytes: bytes, settings: dict) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    arr = np.array(image)

    # Brightness and contrast
    image = ImageEnhance.Brightness(image).enhance(1.0 + settings.get("brightness", 0) / 100)
    image = ImageEnhance.Contrast(image).enhance(1.0 + settings.get("contrast", 0) / 100)

    # Fog
    fog_strength = settings.get("fog", 0) / 100
    if fog_strength > 0:
        overlay = Image.new("RGBA", image.size, (220, 220, 220, int(90 * fog_strength)))
        image = Image.alpha_composite(image.convert("RGBA"), overlay)

    # Rain
    rain_strength = settings.get("rain", 0) / 100
    if rain_strength > 0:
        arr = np.array(image.convert("RGB"))
        height, width, _ = arr.shape
        for _ in range(max(1, int(width * height * rain_strength * 0.0025))):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            length = random.randint(5, 18)
            for i in range(length):
                if y + i < height:
                    arr[y + i, x] = np.clip(arr[y + i, x] + 65, 0, 255)
        image = Image.fromarray(arr.astype("uint8"), mode="RGB")

    # Glare
    glare_strength = settings.get("glare", 0) / 100
    if glare_strength > 0:
        glare = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = glare
        for idx in range(3):
            # transparent white glow to mimic lens flare
            white = Image.new("RGBA", image.size, (255, 255, 255, 0))
            white = white.filter(ImageFilter.GaussianBlur(radius=32 + idx * 8))
        image = Image.alpha_composite(image.convert("RGBA"), glare)

    # Motion blur
    blur_strength = settings.get("blur", 0)
    if blur_strength > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_strength / 20))

    # Noise
    noise_strength = settings.get("noise", 0) / 100
    if noise_strength > 0:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        noise = np.random.normal(0, 25 * noise_strength, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        image = Image.fromarray(arr.astype("uint8"), mode="RGB")

    # Quality loss
    quality = max(20, 100 - int(settings.get("quality", 100) * 0.6))
    buffer = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def reset_image(image_bytes: bytes) -> bytes:
    return image_bytes
