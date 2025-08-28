import argparse
import os
import inspect
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

import torch
import groundingdino
from groundingdino.util.inference import load_model
import groundingdino.datasets.transforms as T  # official transforms
from groundingdino.util.misc import nested_tensor_from_tensor_list
import traceback


# ---------------- device helpers ----------------
def select_device(arg: str) -> str:
  if arg == "mps":
    return "mps"
  if arg == "cpu":
    return "cpu"
  return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------- config path from installed package ----------------
def find_config_path() -> str:
  pkg_dir = Path(inspect.getfile(groundingdino)).parent
  cfg = pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
  if not cfg.exists():
    raise FileNotFoundError(f"Could not find GroundingDINO_SwinT_OGC.py in {pkg_dir}/config")
  return str(cfg)


# ---------------- PSD flatten (uses composite()) ----------------
def flatten_psd_to_bgr(psd_path: Path) -> np.ndarray:
  psd = PSDImage.open(psd_path)
  pil_img = psd.composite()  # RGBA or RGB
  if pil_img.mode == "RGBA":
    bg = Image.new("RGB", pil_img.size, (255, 255, 255))
    bg.paste(pil_img, mask=pil_img.split()[-1])
    pil_img = bg
  else:
    pil_img = pil_img.convert("RGB")
  return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------- tiny post-processing utils ----------------
def boxes_cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
  cx, cy, w, h = cxcywh.unbind(-1)
  x1 = cx - 0.5 * w
  y1 = cy - 0.5 * h
  x2 = cx + 0.5 * w
  y2 = cy + 0.5 * h
  return torch.stack([x1, y1, x2, y2], dim=-1)


def draw_boxes(img_bgr: np.ndarray, boxes_xyxy_norm: np.ndarray, scores, phrases, color=(0, 255, 0)) -> np.ndarray:
  out = img_bgr.copy()
  h, w = out.shape[:2]
  for i in range(len(phrases)):
    x1n, y1n, x2n, y2n = boxes_xyxy_norm[i]
    x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    label = f"{phrases[i]} {float(scores[i]):.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ytext = max(0, y1 - 4)
    cv2.rectangle(out, (x1, ytext - th - 6), (x1 + tw + 6, ytext), color, -1)
    cv2.putText(out, label, (x1 + 3, ytext - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
  return out


def predict_no_cuda(model, image_rgb: np.ndarray, prompt: str, box_threshold: float, device: str):
    if not isinstance(prompt, str) or not prompt.strip():
        return np.zeros((0,4), np.float32), np.array([], np.float32), []

    # normalize prompt like the official helper does
    prompt = prompt.strip()
    if not prompt.endswith("."):
        prompt = prompt + "."

    # --- Resize to long-side=800 ---
    h, w = image_rgb.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((0,4), np.float32), np.array([], np.float32), []
    scale = 800.0 / max(h, w)
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    img_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # --- To tensor + normalize ---
    x = torch.from_numpy(img_resized.copy()).float() / 255.0   # HWC
    x = x.permute(2, 0, 1).contiguous()                        # CHW
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x = (x - mean) / std                                       # still CHW

    # Build NestedTensor (list of tensors -> pads + mask internally)
    samples = nested_tensor_from_tensor_list([x]).to(device)   # <— KEY FIX

    model.eval()
    with torch.no_grad():
        outputs = model(samples, captions=[prompt])            # <— pass NestedTensor

    logits = outputs.get("pred_logits", None)
    boxes  = outputs.get("pred_boxes", None)
    if logits is None or boxes is None or logits.numel() == 0:
        return np.zeros((0,4), np.float32), np.array([], np.float32), []

    logits = logits.sigmoid()[0]                 # (Q, D)
    scores = logits.max(dim=-1).values           # (Q,)
    keep = scores > box_threshold
    if keep.sum().item() == 0:
        return np.zeros((0,4), np.float32), np.array([], np.float32), []

    scores = scores[keep]
    pred_boxes = boxes[0][keep]                  # (K, 4) normalized cxcywh
    if pred_boxes.numel() == 0:
        return np.zeros((0,4), np.float32), np.array([], np.float32), []

    cx, cy, bw, bh = pred_boxes.unbind(-1)
    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    return boxes_xyxy.cpu().numpy(), scores.cpu().numpy(), [prompt] * boxes_xyxy.shape[0]

# ---------------- model loader ----------------
def load_dino(checkpoint_path: str, device: str):
  cfg = find_config_path()
  model = load_model(cfg, checkpoint_path)
  model.to(device).eval()
  return model


# ---------------- main folder processing ----------------
def process_folder(prompt: str, box_thr: float, text_thr: float, input_dir: Path, out_dir: Path,
                   include_images: bool, checkpoint_path: str, device: str):
  out_dir.mkdir(parents=True, exist_ok=True)

  # Disable CUDA probing entirely
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  model = load_dino(checkpoint_path, device)

  exts = ["*.psd", "*.PSD"]
  if include_images:
    exts += ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
  files = []
  for pat in exts:
    files.extend(sorted(input_dir.glob(pat)))

  print(f"Device: {device}")
  print(f"Files to run: {len(files)} | Prompt: '{prompt}' | box_thr={box_thr}, text_thr={text_thr}")

  if not files:
    print("No PSDs/images found.")
    return

  for src in files:
    try:
        if src.suffix.lower() == ".psd":
            img_bgr = flatten_psd_to_bgr(src)
        else:
            img_bgr = cv2.imread(str(src))
            if img_bgr is None:
                raise RuntimeError("Failed to read image")

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"   -> {src.name} | shape={rgb.shape}")  

        boxes, scores, phrases = predict_no_cuda(
            model,
            rgb,
            prompt=prompt,
            box_threshold=box_thr,
            device=device
        )

        print(f"   <- detections: {len(phrases)}")      
        annotated = draw_boxes(img_bgr, boxes, scores, phrases)
        out_path = out_dir / f"{src.stem}_dino.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[ok] {src.name} -> {out_path.name} | detections={len(phrases)}")
    except Exception as e:
        print(f"[ERR] {src.name}: {e}")
        traceback.print_exc(limit=4)     


def main():
  ap = argparse.ArgumentParser("Run GroundingDINO on all PSDs in a folder and save annotated JPEGs.")
  ap.add_argument("--prompt", required=True, help='e.g. "bra", "panties", "t-shirt"')
  ap.add_argument("--weights", required=True, help="Path to groundingdino_swint_ogc.pth")
  ap.add_argument("--box-threshold", type=float, default=0.30)
  ap.add_argument("--text-threshold", type=float, default=0.25)  # kept for API symmetry
  ap.add_argument("--input", default="input", help="Input directory")
  ap.add_argument("--out", default="out", help="Output directory")
  ap.add_argument("--include-images", action="store_true", help="Also process JPG/PNG")
  ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
  args = ap.parse_args()

  device = select_device(args.device)
  process_folder(args.prompt, args.box_threshold, args.text_threshold, Path(args.input),
                 Path(args.out), args.include_images, args.weights, device)


if __name__ == "__main__":
  main()
