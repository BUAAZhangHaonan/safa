from __future__ import annotations


def train_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def train_transform_strong(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.25),
        ]
    )


def eval_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def generator_image_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class R14JointResizeTransform:
    """Resize image, exact bbox mask and 68 points with one shared geometry."""

    def __init__(self, image_size: int, horizontal_flip_probability: float = 0.0):
        if type(image_size) is not int or image_size <= 0:
            raise ValueError("R14 image_size must be a positive integer")
        if type(horizontal_flip_probability) not in {int, float}:
            raise ValueError("R14 horizontal_flip_probability must be numeric")
        probability = float(horizontal_flip_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("R14 horizontal_flip_probability must be in [0, 1]")
        self.image_size = image_size
        self.horizontal_flip_probability = probability

    def __call__(self, image, bbox_xywh, landmarks68):
        import torch
        from PIL import Image, ImageDraw
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as functional

        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("R14 joint transform requires an RGB PIL image")
        if len(bbox_xywh) != 4:
            raise ValueError("R14 joint transform bbox must contain four entries")
        if len(landmarks68) != 68:
            raise ValueError("R14 joint transform requires exactly 68 landmarks")
        original_width, original_height = image.size
        if original_width <= 0 or original_height <= 0:
            raise ValueError("R14 joint transform requires a non-empty image")
        x, y, width, height = (int(value) for value in bbox_xywh)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("R14 joint transform received an invalid bbox")
        if x + width > original_width or y + height > original_height:
            raise ValueError("R14 joint transform bbox exceeds image bounds")

        original_mask = Image.new("L", image.size, 0)
        # Pillow's rectangle endpoint is inclusive. Subtract one so this is
        # exactly the half-open CSV box [x, x+w) x [y, y+h).
        ImageDraw.Draw(original_mask).rectangle(
            (x, y, x + width - 1, y + height - 1),
            fill=255,
        )
        output_size = [self.image_size, self.image_size]
        resized_image = functional.resize(
            image,
            output_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        resized_mask = functional.resize(
            original_mask,
            output_size,
            interpolation=InterpolationMode.NEAREST,
        )

        flip = False
        if self.horizontal_flip_probability > 0.0:
            flip = bool(torch.rand(()) < self.horizontal_flip_probability)
        if flip:
            resized_image = functional.hflip(resized_image)
            resized_mask = functional.hflip(resized_mask)

        image_tensor = functional.pil_to_tensor(resized_image).to(dtype=torch.float32).div_(255.0)
        face_mask = functional.pil_to_tensor(resized_mask).to(dtype=torch.bool)
        nonzero = face_mask[0].nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            raise ValueError("R14 exact bbox vanished during resize")
        min_y, min_x = nonzero.min(dim=0).values
        max_y, max_x = nonzero.max(dim=0).values
        transformed_bbox = torch.tensor(
            [
                float(min_x),
                float(min_y),
                float(max_x - min_x + 1),
                float(max_y - min_y + 1),
            ],
            dtype=torch.float32,
        )

        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height
        landmarks = torch.tensor(landmarks68, dtype=torch.float32)
        landmarks[:, 0].mul_(scale_x)
        landmarks[:, 1].mul_(scale_y)
        if flip:
            landmarks[:, 0] = (self.image_size - 1.0) - landmarks[:, 0]
        if not torch.isfinite(landmarks).all():
            raise ValueError("R14 transformed landmarks contain a non-finite value")
        return {
            "image": image_tensor,
            "face_mask": face_mask,
            "bbox_xywh": transformed_bbox,
            "landmarks68": landmarks,
        }


def r14_joint_transform(image_size: int, horizontal_flip_probability: float = 0.0):
    return R14JointResizeTransform(image_size, horizontal_flip_probability)
