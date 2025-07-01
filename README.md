# AODSAR

**AODSAR** (Automatic Object Detection, Segmentation And Replacement) is an interactive Streamlit application for image inpainting using SAM (Segment Anything Model) and various Stable Diffusion pipelines.

## 🚀 Features

- 📤 Upload an image
- 🔲 Draw bounding boxes on the image
- ✂️ Segment the selected region using [SAM](https://segment-anything.com/)
- 🎨 Inpaint the region using:
  - Kandinsky 2.2 Decoder
  - Stable Diffusion 1
  - Stable Diffusion 2
  - Stable Diffusion v1.5
- ⚡ Refine results with upscaling and a separate refiner model
- 📥 Download inpainted results or enhance them further

---

## 🖼️ Example Workflow

1. Upload your image
2. Draw a bounding box around the object to replace
3. Choose an inpainting model and write a prompt
4. Run the inpainting pipeline
5. (Optional) Enhance the output image with a refiner
6. Download the result

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aodsar.git
cd aodsar

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
aodsar/
├── AODSAR.PY.py              # Main Streamlit UI logic
├── inpaints.py               # Inpainting logic using diffusers
├── sam2.py                   # SAM-based segmentation logic
├── models/                   # Pretrained model weights (e.g., sam2.1_l.pt)
└── assets/
    ├── uploaded images/
    ├── original images/
    ├── masked images/
    ├── segmented images/
    └── refined images/
```

---

## 🧠 Model Setup (IMPORTANT)

Before running the app, you **must download the inpainting models manually** and place them into the `inpaint models/` directory. You can obtain them from:

- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [Stable Diffusion 2](https://huggingface.co/stabilityai/stable-diffusion-2)
- [Kandinsky 2.2 Decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)
- [SAM (Segment Anything)](https://github.com/facebookresearch/segment-anything) - download `sam2.1_l.pt` and place it in the `models/` directory

or simply go to https://drive.google.com/drive/folders/1NFPmSn-MWy6_1iC8g3wz_EUSik4E9_mA?usp=sharing 
and download the inpaint models folder and models folder

> Make sure the directory structure looks like this:
```
inpaint models/
├── Stable Diffusion 1/
├── Stable Diffusion 2/
├── Stable Diffusion-v1-5/
├── Kandinsky-2-2-Decoder/
models/
└── sam2.1_l.pt
```

---

## 🛠️ Usage

Once the models are downloaded and placed correctly:

```bash
streamlit run AODSAR.PY.py
```

The UI will open in your default web browser.

---

## 💬 Prompts

Enter prompts like:
- `Replace with a red sofa`
- `Add a tree in the background`
- `Make the object a cartoon`

Optionally, use a **negative prompt** to avoid unwanted artifacts:
- `blur, distorted, noisy`

---

## ⚠️ Notes

- This app uses GPU (CUDA) for inference. Ensure proper device setup.
- File cleanup and memory handling (via `gc` and CUDA cache clearing) are built-in.
- If running on CPU, some model features (like `xformers`) may be disabled.

---

## 📃 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

- [Meta AI's SAM](https://github.com/facebookresearch/segment-anything)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Streamlit](https://streamlit.io/)
```
