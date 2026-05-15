from __future__ import annotations

from io import BytesIO


def apply_jpeg(images, quality: int):
    from PIL import Image
    import torch
    from torchvision.transforms import functional as TF

    if not (1 <= int(quality) <= 100):
        raise ValueError(f"JPEG quality must be in [1,100], got {quality}")
    output = []
    for image in images.detach().cpu():
        pil = TF.to_pil_image(image.clamp(0, 1))
        buffer = BytesIO()
        pil.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        output.append(TF.to_tensor(Image.open(buffer).convert("RGB")))
    return torch.stack(output, dim=0).to(images.device, dtype=images.dtype)


def apply_blur(images, radius: float):
    from PIL import ImageFilter
    import torch
    from torchvision.transforms import functional as TF

    output = []
    for image in images.detach().cpu():
        pil = TF.to_pil_image(image.clamp(0, 1)).filter(ImageFilter.GaussianBlur(radius=float(radius)))
        output.append(TF.to_tensor(pil))
    return torch.stack(output, dim=0).to(images.device, dtype=images.dtype)


def apply_downsample(images, scale: float):
    import torch.nn.functional as F

    if not (0.0 < float(scale) < 1.0):
        raise ValueError(f"Downsample scale must be in (0,1), got {scale}")
    height, width = images.shape[-2:]
    low_h = max(1, int(height * float(scale)))
    low_w = max(1, int(width * float(scale)))
    low = F.interpolate(images, size=(low_h, low_w), mode="bilinear", align_corners=False)
    return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False).clamp(0, 1)


def apply_center_crop_resize(images, crop_fraction: float):
    import torch.nn.functional as F

    if not (0.0 < float(crop_fraction) <= 1.0):
        raise ValueError(f"Crop fraction must be in (0,1], got {crop_fraction}")
    _, _, height, width = images.shape
    crop_h = max(1, int(height * float(crop_fraction)))
    crop_w = max(1, int(width * float(crop_fraction)))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    crop = images[:, :, top : top + crop_h, left : left + crop_w]
    return F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False).clamp(0, 1)


def apply_noise(images, noise_std: float, seed: int):
    import torch

    if float(noise_std) < 0.0:
        raise ValueError(f"Noise std must be non-negative, got {noise_std}")
    generator = torch.Generator(device=images.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(images.shape, device=images.device, dtype=images.dtype, generator=generator) * float(noise_std)
    return (images + noise).clamp(0, 1)


def perturbation_map(config: dict, seed: int):
    return {
        "jpeg": lambda images: apply_jpeg(images, int(config["jpeg_quality"])),
        "blur": lambda images: apply_blur(images, float(config["blur_radius"])),
        "downsample": lambda images: apply_downsample(images, float(config["downsample_scale"])),
        "crop": lambda images: apply_center_crop_resize(images, float(config["crop_fraction"])),
        "noise": lambda images: apply_noise(images, float(config["noise_std"]), seed),
    }

