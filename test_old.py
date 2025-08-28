import argparse
import os
from pathlib import Path
import inspect

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

import torch
import groundingdino
from groundingdino.util.inference import load_model, predict


def find_config_path() -> str:
  # Use the config bundled in the pip package
  pkg_dir = Path(inspect.getfile(groundingdino)).parent
  cfg = pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
  if not cfg.exists():
    raise FileNotFoundError(f"Could not find GroundingDINO_SwinT_OGC.py in {pkg_dir}/config")
  return str(cfg)


def flatten_psd_to_bgr(psd_path: Path) -> np.ndarray:
  """Flatten visible PSD layers to a white background and return OpenCV BGR ndarray."""
  psd = PSDImage.open(psd_path)
  composite = psd.composite()  # PIL Image RGBA/RGB
  if composite.mode == "RGBA":
    bg = Image.new("RGB", composite.size, (255, 255, 255))
    bg.paste(composite, mask=composite.split()[-1])
    pil_rgb = bg
  else:
    pil_rgb = composite.convert("RGB")
  return cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)


def draw_boxes(img_bgr: np.ndarray, boxes_xyxy_norm: np.ndarray, logits, phrases) -> np.ndarray:
  """Draw normalized xyxy boxes on a BGR image."""
  out = img_bgr.copy()
  h, w = out.shape[:2]
  for i in range(len(boxes_xyxy_norm)):
    x1n, y1n, x2n, y2n = boxes_xyxy_norm[i]
    x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{phrases[i]} {float(logits[i]):.2f}"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(0, y1 - 4)
    cv2.rectangle(out, (x1, y_text - th - 6), (x1 + tw + 6, y_text), (0, 255, 0), -1)
    cv2.putText(out, label, (x1 + 3, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
  return out


def load_dino(checkpoint_path: str, device: str):
  cfg = find_config_path()
  model = load_model(cfg, checkpoint_path)
  model.to(device).eval()
  return model


def process_folder(prompt: str, box_thr: float, text_thr: float, in_dir: Path, out_dir: Path,
                   include_images: bool, checkpoint_path: str, device: str):
  out_dir.mkdir(parents=True, exist_ok=True)
  model = load_dino(checkpoint_path, device)

  exts = ["*.psd", "*.PSD"]
  if include_images:
    exts += ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

  files = []
  for pat in exts:
    files.extend(sorted(Path(in_dir).glob(pat)))

  if not files:
    print("No PSDs (or images) found in current folder.")
    return

  print(f"Device: {device}")
  print(f"Files to run: {len(files)} | Prompt: '{prompt}' | box_thr={box_thr}, text_thr={text_thr}")

  for src in files:
    try:
      if src.suffix.lower() == ".psd":
        img_bgr = flatten_psd_to_bgr(src)
      else:
        img_bgr = cv2.imread(str(src))
        if img_bgr is None:
          raise RuntimeError("Failed to read image")

      # predict expects RGB np array
      boxes, logits, phrases = predict(
          model,
          cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
          caption=prompt,
          box_threshold=box_thr,
          text_threshold=text_thr
      )

      annotated = draw_boxes(img_bgr, boxes, logits, phrases)

      # Save as JPEG next to out/
      out_name = src.stem + "_dino.jpg"
      out_path = out_dir / out_name
      cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

      print(f"[ok] {src.name} -> {out_path.name} | detections={len(phrases)}")
    except Exception as e:
      print(f"[ERR] {src.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
      description="Run GroundingDINO on all PSDs in the current folder and save annotated JPEGs."
  )
  parser.add_argument("--prompt", required=True, help='Text prompt, e.g. "bra" or "panties"')
  parser.add_argument("--weights", required=True, help="Path to groundingdino_swint_ogc.pth")
  parser.add_argument("--box-threshold", type=float, default=0.30, help="Box score threshold")
  parser.add_argument("--text-threshold", type=float, default=0.25, help="Text score threshold")
  parser.add_argument("--input", default="input", help="Input directory")
  parser.add_argument("--out", default="out", help="Output directory")
  parser.add_argument("--include-images", action="store_true",
                      help="Also process JPG/PNG alongside PSDs")
  parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto",
                      help="Force device (default auto = mps if available else cpu)")
  args = parser.parse_args()

  if args.device == "mps":
    device = "mps"
  elif args.device == "cpu":
    device = "cpu"
  else:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

  process_folder(
      prompt=args.prompt,
      box_thr=args.box_threshold,
      text_thr=args.text_threshold,
      in_dir=Path(args.input),
      out_dir=Path(args.out),
      include_images=args.include_images,
      checkpoint_path=args.weights,
      device=device
  )


if __name__ == "__main__":
  main()
