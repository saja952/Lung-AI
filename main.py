# main.py
import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import torch
import cv2
from model import DAR_UNet, AttentionUNetFromWeights, AttentionUNetSmallCheckpoint


st.set_page_config(page_title="Lung AI - Segmentation", page_icon="🫁", layout="wide")


MODEL_FILES = {
    "Attention U-Net (small)": ("best_attention_unet.pth", "att_small"),
    "U-Net (baseline)": ("best_model.pth", "att_big"),
    "DAR-UNet": ("BestModel_DAR.pth", "dar"),
}

LOGO_PATH = "pic.jpeg"
DEFAULT_SIZE = 256


def ensure_file_exists(path: str):
    if not os.path.exists(path):
        st.error(
            f"File not found: {path}\n\n"
            f" Put it in the same folder as main.py, or update the path in MODEL_FILES."
        )
        st.stop()


def preprocess_image(pil_img: Image.Image, target_size: int, in_channels: int):

    img = pil_img.convert("RGB").resize((target_size, target_size))
    img_rgb = np.array(img)

    if in_channels == 1:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        x = gray[None, :, :]
    else:
        x = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

    x = torch.from_numpy(x).unsqueeze(0)
    return x, img_rgb


@torch.no_grad()
def predict_mask(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    logits = model(x)  # expected [1,1,H,W]
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # H,W
    return probs


def make_overlay(img_rgb: np.ndarray, mask_bin: np.ndarray, alpha: float) -> np.ndarray:
    overlay = img_rgb.copy()
    color = np.zeros_like(img_rgb)
    color[:, :, 1] = 255  # green mask
    mask3 = np.repeat(mask_bin[:, :, None], 3, axis=2)
    overlay = np.where(mask3 == 1, (1 - alpha) * overlay + alpha * color, overlay).astype(np.uint8)
    return overlay


def load_state_dict_any(path: str):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


@st.cache_resource
def build_and_load_model(model_path: str, kind: str, device_str: str):

    device = torch.device(device_str)

    if kind == "dar":
        model = DAR_UNet(in_channels=3, out_channels=1, dropout_rate=0.0)
    elif kind == "att_small":
        model = AttentionUNetSmallCheckpoint(in_channels=1, out_channels=1, base=32)
    else:
        model = AttentionUNetFromWeights(in_channels=3, out_channels=1, base=32)

    state = load_state_dict_any(model_path)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model


st.sidebar.title("Settings")

choice = st.sidebar.selectbox("Choose Model", list(MODEL_FILES.keys()))
model_path, model_kind = MODEL_FILES[choice]

img_size = st.sidebar.select_slider("Image Size", options=[128, 256, 384, 512], value=DEFAULT_SIZE)
threshold = st.sidebar.slider("Mask Threshold", 0.10, 0.90, 0.50, 0.01)
show_overlay = st.sidebar.checkbox("Show Overlay", True)
overlay_alpha = st.sidebar.slider("Overlay Alpha", 0.10, 0.90, 0.45, 0.01)

device_str = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Device: **{device_str.upper()}**")


c1, c2 = st.columns([1, 3])

with c1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.info("Put your logo as pic.jpeg (or change LOGO_PATH).")

with c2:
    st.markdown("## Lung AI — Lung Segmentation")
    st.write("Upload a chest X-ray image, select a model, and generate the lung mask.")

st.divider()


ensure_file_exists(model_path)



try:
    model = build_and_load_model(model_path, model_kind, device_str=device_str)
except Exception as e:
    st.error(f" Model load failed for: {choice}\n\n{e}")
    st.stop()

st.caption(f"Selected model: **{choice}**  |  Weights: `{model_path}`")


uploaded = st.file_uploader("Upload an X-ray image (PNG/JPG)", type=["png", "jpg", "jpeg"])
run_btn = st.button("Run Segmentation", type="primary", disabled=(uploaded is None))

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

pil_img = Image.open(io.BytesIO(uploaded.read()))

in_ch = 1 if model_kind == "att_small" else 3
x, img_rgb = preprocess_image(pil_img, img_size, in_channels=in_ch)

left, right = st.columns(2)
with left:
    st.markdown("### Input")
    st.image(img_rgb, use_container_width=True)

with right:
    if not run_btn:
        st.info("Click **Run Segmentation** to see results.")

if run_btn:
    with st.spinner("Running inference..."):
        x = x.to(torch.device(device_str))
        probs = predict_mask(model, x)
        mask_bin = (probs >= threshold).astype(np.uint8)

    with right:
        st.markdown("### Output")
        st.write("Mask (binary)")
        st.image((mask_bin * 255).astype(np.uint8), use_container_width=True)

        if show_overlay:
            st.write("Overlay")
            overlay = make_overlay(img_rgb, mask_bin, overlay_alpha)
            st.image(overlay, use_container_width=True)


