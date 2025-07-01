import cv2
import numpy as np
import torch
from ultralytics import SAM
from pathlib import Path
from PIL import Image
import gc
from torch import cuda
import os
import random

class SAMSegmenter:
    def __init__(self, image_path, bbox):
        self.image_path = image_path
        self.base_name = Path(image_path).stem
        self._prepare_dirs()
        self.rand_id = self._generate_random_id()
        self.image_resized, self.image_pil = self._load_and_resize_image()
        self.bbox = bbox
        if not self.bbox:
            raise ValueError("Bounding box must be provided.")
        self.mask_pil, self.overlay_pil = self._run_segmentation()

    def _prepare_dirs(self):
        os.makedirs("assets/original images", exist_ok=True)
        os.makedirs("assets/masked images", exist_ok=True)
        os.makedirs("assets/segmented images", exist_ok=True)

    def _generate_random_id(self):
        for _ in range(100):
            rand_num = random.randint(1, 100)
            filenames = [
                f"assets/original images/{self.base_name}_{rand_num}.png",
                f"assets/masked images/{self.base_name}_{rand_num}.png",
                f"assets/segmented images/{self.base_name}_{rand_num}.png"
            ]
            if not any(os.path.exists(f) for f in filenames):
                return rand_num
        raise RuntimeError("Unable to generate unique random number after 100 attempts.")

    def _load_and_resize_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")
        image_resized = cv2.resize(image, (512, 512))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb).convert("RGB")
        save_path = f"assets/original images/{self.base_name}_{self.rand_id}.png"
        image_pil.save(save_path)
        print(f"Saved resized input image as {save_path}")
        return image_resized, image_pil

    def _run_segmentation(self):
        model = SAM("models/sam2.1_l.pt")
        results = model(self.image_resized, bboxes=[self.bbox])

        print("Model running on:", next(model.model.parameters()).device)

        mask_tensor = results[0].masks.data[0]
        mask = mask_tensor.cpu().numpy().astype(np.uint8)

        mask_pil = Image.fromarray(mask * 255).convert("RGB")
        mask_path = f"assets/masked images/{self.base_name}_{self.rand_id}.png"
        mask_pil.save(mask_path)
        print(f"Saved binary mask to {mask_path}")

        overlay = self.image_resized.copy()
        overlay[mask == 1] = [0, 0, 255]
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay_pil = Image.fromarray(overlay_rgb).convert("RGB")
        overlay_path = f"assets/segmented images/{self.base_name}_{self.rand_id}.png"
        overlay_pil.save(overlay_path)
        print(f"Saved segmented overlay to {overlay_path}")

        del model, results, mask_tensor, mask, overlay
        gc.collect()
        if torch.cuda.is_available():
            cuda.empty_cache()
            cuda.ipc_collect()
        print("✅ GPU and RAM memory cleared.")

        return mask_pil, overlay_pil

    def get_images(self):
        return self.mask_pil, self.image_pil, self.overlay_pil
