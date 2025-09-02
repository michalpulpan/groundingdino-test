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


# ---------------- geometry helpers ----------------
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def clamp_box_xyxy(x1, y1, x2, y2, w, h):
    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 1, w)
    y2 = clamp(y2, 1, h)
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def largest_fit_aspect(w, h, aspect):
    """Largest box of 'aspect' inside WxH, centered."""
    img_ratio = w / h
    if img_ratio > aspect:
        crop_h = h
        crop_w = int(round(aspect * crop_h))
    else:
        crop_w = w
        crop_h = int(round(crop_w / aspect))
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    return x1, y1, x1 + crop_w, y1 + crop_h

def expand_box_with_padding(x1, y1, x2, y2, pad_x_frac, pad_y_frac, w, h):
    """Expand a box by a fraction of its size, clamped to image bounds."""
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * pad_x_frac))
    pad_y = int(round(bh * pad_y_frac))
    x1e = x1 - pad_x
    y1e = y1 - pad_y
    x2e = x2 + pad_x
    y2e = y2 + pad_y
    return clamp_box_xyxy(x1e, y1e, x2e, y2e, w, h)

def draw_final_crop_rectangle(img_bgr, crop_xyxy, color=(0,0,255), thickness=2):
    x1, y1, x2, y2 = map(int, crop_xyxy)
    out = img_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out

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

def union_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)

def expand_norm_box(det_norm, w, h, pad_x_frac, pad_y_frac):
    x1n, y1n, x2n, y2n = det_norm
    x1 = int(round(x1n * w)); y1 = int(round(y1n * h))
    x2 = int(round(x2n * w)); y2 = int(round(y2n * h))
    return expand_box_with_padding(x1, y1, x2, y2, pad_x_frac, pad_y_frac, w, h)


# ---------------- cropping planners ----------------
def plan_primary_crop_4x5(
    img_w, img_h,
    det_box_norm,
    pad_x_frac=0.12, pad_y_frac=0.10,
    aspect=0.8,
    anchor_y=0.50  # 0=top .. 1=bottom position of product center inside the final crop
):
    """
    Minimal 4:5 covering the padded product. Place vertically by anchor_y (subject to bounds).
    """
    x1n, y1n, x2n, y2n = det_box_norm
    x1 = int(round(x1n * img_w))
    y1 = int(round(y1n * img_h))
    x2 = int(round(x2n * img_w))
    y2 = int(round(y2n * img_h))
    x1, y1, x2, y2 = clamp_box_xyxy(x1, y1, x2, y2, img_w, img_h)

    x1e, y1e, x2e, y2e = expand_box_with_padding(x1, y1, x2, y2, pad_x_frac, pad_y_frac, img_w, img_h)
    bw, bh = x2e - x1e, y2e - y1e
    cx, cy = x1e + bw / 2.0, y1e + bh / 2.0

    if bw / bh > aspect:
        crop_w = bw
        crop_h = int(round(crop_w / aspect))
    else:
        crop_h = bh
        crop_w = int(round(crop_h * aspect))

    x1c = int(round(cx - crop_w / 2.0))
    x2c = x1c + crop_w
    target_top = int(round(cy - anchor_y * crop_h))
    y1c = target_top
    y2c = y1c + crop_h

    if x1c < 0:
        x2c -= x1c; x1c = 0
    if x2c > img_w:
        shift = x2c - img_w
        x1c -= shift; x2c = img_w

    lo = max(0, y2e - crop_h)
    hi = min(y1e, img_h - crop_h)
    if lo > hi:
        return largest_fit_aspect(img_w, img_h, aspect)
    y1c = int(clamp(target_top, lo, hi)); y2c = y1c + crop_h

    return clamp_box_xyxy(x1c, y1c, x2c, y2c, img_w, img_h)


def plan_primary_crop_with_head_4x5(
    img_w, img_h,
    det_box_norm,        # product box (normalized)
    head_box_norm=None,  # head/face box (normalized) or None
    pad_x_frac=0.12, pad_y_frac=0.10,
    head_pad_x=0.10, head_pad_y=0.12,
    aspect=0.8, anchor_y=0.90,  # default: push product lower
):
    # expand product
    x1e, y1e, x2e, y2e = expand_norm_box(det_box_norm, img_w, img_h, pad_x_frac, pad_y_frac)
    ux1, uy1, ux2, uy2 = x1e, y1e, x2e, y2e

    # include head if present
    if head_box_norm is not None:
        hx1e, hy1e, hx2e, hy2e = expand_norm_box(head_box_norm, img_w, img_h, head_pad_x, head_pad_y)
        ux1, uy1, ux2, uy2 = union_xyxy((ux1, uy1, ux2, uy2), (hx1e, hy1e, hx2e, hy2e))
        ux1, uy1, ux2, uy2 = clamp_box_xyxy(ux1, uy1, ux2, uy2, img_w, img_h)

    # now compute a crop that covers the union, with placement logic
    bw, bh = max(1, ux2 - ux1), max(1, uy2 - uy1)
    cx, cy = ux1 + bw / 2.0, uy1 + bh / 2.0

    if bw / bh > aspect:
        crop_w = bw
        crop_h = int(round(crop_w / aspect))
    else:
        crop_h = bh
        crop_w = int(round(crop_h * aspect))

    x1c = int(round(cx - crop_w / 2.0))
    x2c = x1c + crop_w
    if x1c < 0:
        x2c -= x1c; x1c = 0
    if x2c > img_w:
        shift = x2c - img_w
        x1c -= shift; x2c = img_w

    target_top = int(round(cy - anchor_y * crop_h))
    lo = max(0, uy2 - crop_h)
    hi = min(img_h - crop_h, uy1)
    if lo > hi:
        return largest_fit_aspect(img_w, img_h, aspect)
    y1c = int(clamp(target_top, lo, hi)); y2c = y1c + crop_h

    return clamp_box_xyxy(x1c, y1c, x2c, y2c, img_w, img_h)


def plan_other_visible_4x5(
    img_w, img_h,
    det_box_norm,
    pad_x_frac=0.12, pad_y_frac=0.10,
    aspect=0.8,
    scale=1.25,          # enlarge crop to show more body
    anchor_y=0.60,       # place product slightly lower than center by default
    head_box_norm=None,  # if show-head, include head in the union
    head_pad_x=0.10, head_pad_y=0.12
):
    # base union = padded product (and head if provided)
    x1e, y1e, x2e, y2e = expand_norm_box(det_box_norm, img_w, img_h, pad_x_frac, pad_y_frac)
    ux1, uy1, ux2, uy2 = x1e, y1e, x2e, y2e
    if head_box_norm is not None:
        hx1e, hy1e, hx2e, hy2e = expand_norm_box(head_box_norm, img_w, img_h, head_pad_x, head_pad_y)
        ux1, uy1, ux2, uy2 = union_xyxy((ux1, uy1, ux2, uy2), (hx1e, hy1e, hx2e, hy2e))
        ux1, uy1, ux2, uy2 = clamp_box_xyxy(ux1, uy1, ux2, uy2, img_w, img_h)

    bw, bh = max(1, ux2 - ux1), max(1, uy2 - uy1)
    cx, cy = ux1 + bw / 2.0, uy1 + bh / 2.0

    # start from minimal cover 4:5
    if bw / bh > aspect:
        crop_w = bw
        crop_h = int(round(crop_w / aspect))
    else:
        crop_h = bh
        crop_w = int(round(crop_h * aspect))

    # scale up to show more body, within image limits
    crop_h = int(round(min(img_h, crop_h * scale)))
    crop_w = int(round(min(img_w, aspect * crop_h)))

    # center X on union
    x1c = int(round(cx - crop_w / 2.0)); x2c = x1c + crop_w
    if x1c < 0: x2c -= x1c; x1c = 0
    if x2c > img_w:
        shift = x2c - img_w
        x1c -= shift; x2c = img_w

    # anchor in Y while keeping union inside
    target_top = int(round(cy - anchor_y * crop_h))
    lo = max(0, uy2 - crop_h)
    hi = min(img_h - crop_h, uy1)
    if lo > hi:
        return largest_fit_aspect(img_w, img_h, aspect)
    y1c = int(clamp(target_top, lo, hi)); y2c = y1c + crop_h

    return clamp_box_xyxy(x1c, y1c, x2c, y2c, img_w, img_h)


# ---------------- prediction ----------------
def predict_no_cuda(model, image_rgb: np.ndarray, prompt: str, box_threshold: float, device: str):
    if not isinstance(prompt, str) or not prompt.strip():
        return np.zeros((0,4), np.float32), np.array([], np.float32), []

    prompt = prompt.strip()
    if not prompt.endswith("."):
        prompt = prompt + "."

    h, w = image_rgb.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((0,4), np.float32), np.array([], np.float32), []
    scale = 800.0 / max(h, w)
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    img_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(img_resized.copy()).float() / 255.0   # HWC
    x = x.permute(2, 0, 1).contiguous()                        # CHW
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x = (x - mean) / std                                       # still CHW

    samples = nested_tensor_from_tensor_list([x]).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(samples, captions=[prompt])

    logits = outputs.get("pred_logits", None)
    boxes  = outputs.get("pred_boxes", None)
    if logits is None or boxes is None or logits.numel() == 0:
        return np.zeros((0,4), np.float32), np.array([], np.float32), []

    logits = logits.sigmoid()[0]
    scores = logits.max(dim=-1).values
    keep = scores > box_threshold
    if keep.sum().item() == 0:
        return np.zeros((0,4), np.float32), np.array([], np.float32), []

    scores = scores[keep]
    pred_boxes = boxes[0][keep]  # cx,cy,w,h normalized
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
                   include_images: bool, checkpoint_path: str, device: str,
                   aspect: float, mode: str, pad_x: float, pad_y: float,
                   out_w: int, out_h: int, anchor_y: float,
                   show_head: bool, head_pad_x: float, head_pad_y: float,
                   other_scale: float, other_anchor_y: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preview").mkdir(parents=True, exist_ok=True)
    (out_dir / "crop").mkdir(parents=True, exist_ok=True)

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
    print(f"Files to run: {len(files)} | Prompt: '{prompt}' | box_thr={box_thr}")

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

            boxes, scores, _ = predict_no_cuda(
                model, rgb, prompt=prompt, box_threshold=box_thr, device=device
            )

            if boxes.size == 0:
                print(f"   !! no detections")
                crop_xyxy = largest_fit_aspect(img_bgr.shape[1], img_bgr.shape[0], aspect)
                preview = draw_final_crop_rectangle(img_bgr, crop_xyxy, (0,0,255), 3)
            else:
                best = int(np.argmax(scores))
                det_norm = boxes[best]  # normalized xyxy

                # optional head detection
                head_norm = None
                if show_head:
                    boxes_h, scores_h, _ = predict_no_cuda(
                        model, rgb, prompt="head.", box_threshold=max(0.30, box_thr), device=device
                    )
                    if boxes_h.size == 0:
                        boxes_h, scores_h, _ = predict_no_cuda(
                            model, rgb, prompt="face.", box_threshold=max(0.30, box_thr), device=device
                        )
                    if boxes_h.size:
                        head_norm = boxes_h[int(np.argmax(scores_h))]

                # pick crop planner
                if mode == "primary":
                    if show_head:
                        crop_xyxy = plan_primary_crop_with_head_4x5(
                            img_bgr.shape[1], img_bgr.shape[0],
                            det_box_norm=det_norm,
                            head_box_norm=head_norm,
                            pad_x_frac=pad_x, pad_y_frac=pad_y,
                            head_pad_x=head_pad_x, head_pad_y=head_pad_y,
                            aspect=aspect, anchor_y=anchor_y
                        )
                    else:
                        crop_xyxy = plan_primary_crop_4x5(
                            img_bgr.shape[1], img_bgr.shape[0],
                            det_box_norm=det_norm,
                            pad_x_frac=pad_x, pad_y_frac=pad_y,
                            aspect=aspect, anchor_y=anchor_y
                        )
                else:  # other
                    crop_xyxy = plan_other_visible_4x5(
                        img_bgr.shape[1], img_bgr.shape[0],
                        det_box_norm=det_norm,
                        pad_x_frac=pad_x, pad_y_frac=pad_y,
                        aspect=aspect, scale=other_scale, anchor_y=other_anchor_y,
                        head_box_norm=head_norm if show_head else None,
                        head_pad_x=head_pad_x, head_pad_y=head_pad_y
                    )

                # preview: detections + final crop
                preview = draw_boxes(img_bgr, boxes, scores, [prompt]*len(scores))  # green boxes
                preview = draw_final_crop_rectangle(preview, crop_xyxy, color=(0,0,255), thickness=3)

            # save preview
            preview_path = out_dir / f"preview/{src.stem}_preview.jpg"
            cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # final crop
            x1, y1, x2, y2 = map(int, crop_xyxy)
            cropped = img_bgr[y1:y2, x1:x2].copy()
            resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            final_path = out_dir / f"crop/{src.stem}_crop_{out_w}x{out_h}.jpg"
            cv2.imwrite(str(final_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            print(f"[ok] {src.name} -> {preview_path.name} (preview), {final_path.name} (crop)")
        except Exception as e:
            print(f"[ERR] {src.name}: {e}")
            traceback.print_exc(limit=4)


def main():
    ap = argparse.ArgumentParser("Run GroundingDINO on all PSDs in a folder and save cropped JPEGs.")
    ap.add_argument("--prompt", required=True, help='e.g. "bra", "panties", "t-shirt"')
    ap.add_argument("--weights", required=True, help="Path to groundingdino_swint_ogc.pth")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)  # kept for API symmetry
    ap.add_argument("--input", default="input", help="Input directory")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--include-images", action="store_true", help="Also process JPG/PNG")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--crop-size", default="2000x2500", help="Output size WxH, e.g. 2000x2500")
    ap.add_argument("--mode", choices=["primary","other"], default="primary",
                    help="primary: use detection (+ optional head) + padding; other: show more body but keep product visible")
    ap.add_argument("--pad-x", type=float, default=0.14, help="Padding fraction of product width")
    ap.add_argument("--pad-y", type=float, default=0.12, help="Padding fraction of product height")
    ap.add_argument("--place", choices=["upper","center","lower"], default="center",
                    help="Where the product sits vertically in PRIMARY crops.")
    ap.add_argument("--anchor-y", type=float, default=None,
                    help="Override --place with explicit 0..1 anchor (0=top, 0.5=center, 1=bottom).")

    # NEW: head control
    ap.add_argument("--show-head", action="store_true",
                    help="Also detect head/face and force the crop to include it (union with product).")
    ap.add_argument("--head-pad-x", type=float, default=0.10, help="Padding for head box (x fraction)")
    ap.add_argument("--head-pad-y", type=float, default=0.12, help="Padding for head box (y fraction)")

    # NEW: secondary images (mode=other) controls
    ap.add_argument("--other-scale", type=float, default=1.25,
                    help="Scale factor over minimal 4:5 to show more body (mode=other).")
    ap.add_argument("--other-anchor-y", type=float, default=0.60,
                    help="Vertical anchor for mode=other (0=top..1=bottom).")

    args = ap.parse_args()

    out_w, out_h = map(int, args.crop_size.lower().split("x"))
    aspect = out_w / out_h

    place_to_anchor = {"upper": 0.15, "center": 0.50, "lower": 0.90}
    anchor_y = args.anchor_y if args.anchor_y is not None else place_to_anchor[args.place]

    device = select_device(args.device)
    process_folder(
        prompt=args.prompt,
        box_thr=args.box_threshold,
        text_thr=args.text_threshold,
        input_dir=Path(args.input),
        out_dir=Path(args.out),
        include_images=args.include_images,
        checkpoint_path=args.weights,
        device=device,
        aspect=aspect,
        mode=args.mode,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        out_w=out_w,
        out_h=out_h,
        anchor_y=anchor_y,
        show_head=args.show_head,
        head_pad_x=args.head_pad_x,
        head_pad_y=args.head_pad_y,
        other_scale=args.other_scale,
        other_anchor_y=args.other_anchor_y,
    )


if __name__ == "__main__":
    main()