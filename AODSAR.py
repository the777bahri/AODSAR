from sam2 import SAMSegmenter
from inpaints import Inpainter
from pathlib import Path
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import io
from diffusers import AutoPipelineForInpainting
import torch


st.title("AODSAR")
st.title("Inpainting with SAM2 and Stable Diffusion")

# Upload image
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.warning("Please upload an image.")
    st.stop()

# Save uploaded file locally
upload_path = os.path.join("assets", "uploaded images", uploaded.name)
os.makedirs(os.path.dirname(upload_path), exist_ok=True)
with open(upload_path, "wb") as f:
    f.write(uploaded.getbuffer())

# Load and resize image for drawing
img = Image.open(upload_path).resize((512, 512))

st.markdown("**Uploaded Image**")
st.image(img, use_column_width=True)

st.markdown("**Draw Bounding Box**")
canvas_res = st_canvas(
    background_image=img,
    update_streamlit=True,
    drawing_mode="rect",
    stroke_width=2,
    stroke_color="#00FF00",
    height=img.height,
    width=img.width,
    key="canvas",
)

# Get bounding box
bbox = None
if canvas_res.json_data and canvas_res.json_data["objects"]:
    obj = canvas_res.json_data["objects"][0]
    left = int(obj["left"])
    top = int(obj["top"])
    width = int(obj["width"])
    height = int(obj["height"])
    bbox = [left, top, left + width, top + height]
    st.success(f"Bounding box: {bbox}")

# Segment only once and store results
if bbox and "segmenter_rand_id" not in st.session_state:
    segmenter = SAMSegmenter(upload_path, bbox)
    mask_img, original_img, segmented_img = segmenter.get_images()

    st.session_state["segmenter_rand_id"] = segmenter.rand_id
    st.session_state["seg_mask"] = mask_img
    st.session_state["seg_orig"] = original_img
    st.session_state["seg_segmented"] = segmented_img

# Display segmentation results 
if "seg_mask" in st.session_state:
    st.subheader("Segmentation Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original Image**")
        st.image(st.session_state["seg_orig"], use_column_width=True)
    with col2:
        st.markdown("**Masked Image**")
        st.image(st.session_state["seg_mask"], use_column_width=True)
    with col3:
        st.markdown("**Segmented Image**")
        st.image(st.session_state["seg_segmented"], use_column_width=True)

# -------------------------
# Inpainting Section
# -------------------------
st.divider()
st.subheader("Inpainting")

# Prompt inputs
prompt = st.text_input("Prompt", placeholder="e.g. Replace with a red sofa")
negative_prompt = st.text_input("Negative Prompt (optional)", placeholder="e.g. blur, distorted")

# Sliders
guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, step=0.5)
strength = st.slider("Strength", 0.1, 1.0, 0.5, step=0.01)
num_images = st.slider("Number of Images", 1, 3, 1)
steps = st.slider("Steps", 10, 100, 50, step=1)

# Model options (single selection)
model_options = {
    "1": "Kandinsky-2-2-Decoder",
    "2": "Stable Diffusion 1",
    "3": "Stable Diffusion 2",
    "4": "Stable Diffusion-v1-5"
}
selected_model = st.selectbox("Select inpainting model", options=list(model_options.keys()),
                              format_func=lambda k: model_options[k])

# Run inpainting
if st.button("Run Inpainting"):
    if not prompt:
        st.error("Prompt is required.")
    elif "segmenter_rand_id" not in st.session_state:
        st.error("No segmentation result found. Draw a bounding box first.")
    else:
        with st.spinner("Generating inpainted images..."):
            rand_id = st.session_state["segmenter_rand_id"]
            base_stem = Path(upload_path).stem
            original_path = os.path.abspath(f"assets/original images/{base_stem}_{rand_id}.png")
            mask_path = os.path.abspath(f"assets/masked images/{base_stem}_{rand_id}.png")

            inpainter = Inpainter(original_path, mask_path)
            inpainter.load_model(selected_model)

            images = inpainter.inpaint(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                strength=strength,
                num_images=num_images,
                steps=steps
            )

            # Save both images and bytes
            img_bytes_list = []
            for img in images:
                img_io = io.BytesIO()
                img.save(img_io, format="PNG")
                img_io.seek(0)
                img_bytes_list.append(img_io)

            st.session_state["generated_images"] = images
            st.session_state["generated_images_bytes"] = img_bytes_list


def enhance_image(i, img, mask_img, prompt):
   # Resize only a copy of the image to 1024x1024 for enhancement
    img_upscaled = img.copy().resize((1024, 1024))

    # Load refiner pipeline (cache in session to avoid reloading)
    if "refine_pipeline" not in st.session_state:
        refine_pipeline = AutoPipelineForInpainting.from_pretrained(
            "inpaint models/image refiner", local_files_only=True, torch_dtype=torch.float16
        )
        refine_pipeline.enable_model_cpu_offload()
        refine_pipeline.enable_xformers_memory_efficient_attention()
        st.session_state["refine_pipeline"] = refine_pipeline
    else:
        refine_pipeline = st.session_state["refine_pipeline"]

    # Run refinement
    refined = refine_pipeline(
        prompt=prompt,
        image=img_upscaled,
        mask_image=mask_img,
        output_type="pil"
    ).images[0]

    refined = refined.resize((512, 512))
    # Save the refined image to a folder
    output_folder = "assets/refined images"  
    refined.save(os.path.join(output_folder, f"refined_image_{i}.png"))
    # Resize refined result back to 512x512
    return refined



# Display generated images and allow download/enhance
if "generated_images" in st.session_state and "generated_images_bytes" in st.session_state:
    images = st.session_state["generated_images"]
    image_bytes_list = st.session_state["generated_images_bytes"]
    st.markdown(f"### Results: {model_options[selected_model]}")
    cols = st.columns(len(images))
    for i, (col, img, img_bytes) in enumerate(zip(cols, images, image_bytes_list), start=1):
        with col:
            st.image(img, caption=f"Generated Image {i}", use_column_width=True)

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.download_button(
                    label="Download",
                    data=img_bytes,
                    file_name=f"inpainted_image_{i}.png",
                    mime="image/png",
                    key=f"download_{i}"
                )
            
            with btn_col2:
                if st.button("Enhance", key=f"enhance_{i}"):
                    if not prompt:
                        st.error("Please enter a prompt before enhancement.")
                    else:
                        st.info(f"Enhancing image {i}...")

                        # Call enhance and replace image and bytes
                        enhanced = enhance_image(i, img, st.session_state["seg_mask"], prompt)

                        # Update image and bytes in session state
                        st.session_state["generated_images"][i - 1] = enhanced
                        enhanced_io = io.BytesIO()
                        enhanced.save(enhanced_io, format="PNG")
                        enhanced_io.seek(0)
                        st.session_state["generated_images_bytes"][i - 1] = enhanced_io

                        st.success(f"Image {i} enhanced!")
                        st.experimental_rerun()
