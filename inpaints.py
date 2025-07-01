import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import gc
import os
import random
from pathlib import Path
from torch import cuda

class Inpainter:
    def __init__(self, init_image_path, mask_image_path):
        self.init_image_path = init_image_path
        self.mask_image_path = mask_image_path
        self.init_image = load_image(init_image_path)
        self.mask_image = load_image(mask_image_path)
        self.image_stem = Path(init_image_path).stem
        self.models = {
            "1": ("Kandinsky-2-2-Decoder", "inpaint models/Kandinsky-2-2-Decoder"),
            "2": ("Stable Diffusion 1", "inpaint models/Stable Diffusion 1"),
            "3": ("Stable Diffusion 2", "inpaint models/Stable Diffusion 2"),
            "4": ("Stable Diffusion-v1-5", "inpaint models/Stable Diffusion-v1-5")
        }

    def load_model(self, model_number):
        model = self.models.get(model_number)
        if not model:
            raise ValueError(f"Invalid selection. Choose from the list.")

        self.model_key, model_path = model
        print(f"Loading model from: {model_path}")
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16
        )
        self.pipeline.enable_model_cpu_offload()
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            print("xFormers optimization enabled")
        except Exception as e:
            print("xFormers optimization not enabled:", e)

    def _generate_unique_filename(self):
        for _ in range(100):
            rand_num = random.randint(1, 100)
            filename = f"{self.model_key}_{self.image_stem}_{rand_num}.png"
            path = os.path.join("assets\inpainted images", filename)
            if not os.path.exists(path):
                return path
        raise RuntimeError("Unable to generate unique filename after 100 attempts.")

    def inpaint(self, prompt, negative_prompt=None, guidance_scale=None, strength=0.5, num_images=1, seed=92, steps=None, progress_callback=None):
        if not prompt:
            raise ValueError("Prompt is required.")

        generator = torch.Generator("cuda").manual_seed(seed)
        total_steps = steps if steps else 50  # Default to 50 steps if not provided

        results = []
        for current_step in range(1, total_steps + 1):
            if progress_callback:
                progress_callback(current_step, total_steps)

            # Perform the inpainting process
            if current_step == total_steps:
                results = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=self.init_image,
                    mask_image=self.mask_image,
                    guidance_scale=guidance_scale,
                    strength=strength,  
                    generator=generator,
                    num_inference_steps=steps,
                    num_images_per_prompt=num_images
                ).images

        # Save the generated images and return them as a list
        generated_images = []
        for img in results:
            save_path = self._generate_unique_filename()
            img.save(save_path)
            print(f"Saved inpainted image to {save_path}")
            generated_images.append(img) 

        self.cleanup()
        return generated_images  

    def cleanup(self):
        print("\n🧹 Cleaning up GPU and RAM...")

        # Delete class attributes
        if hasattr(self, 'pipeline'):
            del self.pipeline
            self.pipeline = None
        if hasattr(self, 'init_image'):
            del self.init_image
            self.init_image = None
        if hasattr(self, 'mask_image'):
            del self.mask_image
            self.mask_image = None
        if hasattr(self, 'model_key'):
            del self.model_key
            self.model_key = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            cuda.empty_cache()
            cuda.ipc_collect()

        print("✅ Memory cleared.")