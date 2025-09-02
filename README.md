# Groundingdino-test: Text-Driven Object Detection for Fashion Items from PSD

This repository provides a step-by-step guide for running Grounding DINO object detection with text prompts, specifically optimized for fashion item detection. It works on both x86 and ARM-based Mac systems.

## Overview

Grounding DINO combines the power of vision transformers with text encoders to detect objects based on natural language descriptions. This implementation focuses on identifying clothing items like underwear, bras, and leggings in images.

## Features

- Text-prompted object detection
- Support for ARM architecture (M1/M2 Macs)
- Easy-to-follow setup process
- Batch processing of multiple images

## Use Cases

- Batch PSD processing with fashion item detection
- Automated tagging of clothing items
- Dataset preparation for fashion-related machine learning tasks

## Setup (even for arch. ARM MAC)

Note: Make sure to run Python 3.9, 3.13 did not work as of 28/08/2025)

```
python3.9 -m venv venv install 3.11.9
source .venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.2,<2.4" torchvision torchaudio
pip install groundingdino-py psd-tools pillow opencv-pytho
pip install -r requirements.txt
curl -L -o weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

```

## Test commands:

`python test.py --prompt "bottom underware panties." --input ./input/bra --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_panties`

`python test.py --prompt "upper underware bra." --input ./input/bra --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu`

`python test.py --prompt "bottom leggins pants." --input ./input/leggins --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_leggins`

Primary image for bra while still showing head:
`python test.py --prompt "upper underware bra." --input ./input/bra --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_bra --mode primary --show-head --pad-x 0.18 --pad-y 0.16 --head-pad-x 0.14 --head-pad-y 0.25 --place lower --anchor-y 0.93 --box-threshold 0.35`

Primary image for panties/bottom underware:
`python test.py --prompt "bottom underware panties." --input ./input/bra --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_panties --place center --mode primary --pad-x 0.2 --pad-y 0.22 --anchor-y 0.55 --box-threshold 0.35`

Primary image for leggins/pants/shorts:
`python test.py --prompt "bottom leggins pants." --input ./input/leggins --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_leggins --mode primary --place upper --pad-x 0.1 --pad-y 0.2 --box-threshold 0.35`

Primary image for t-shirt/camisole:
`python test.py --prompt "tshirt." --input ./input/tshirt --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_tshirt --mode primary --place lower --show-head --box-threshold 0.25`

Primary image for pyjama trousers:
`python test.py --prompt "bottom pyjama pants." --input ./input/pyjama-trousers --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_pyjama-trousers --mode primary --place upper --pad-x 0.1 --pad-y 0.2 --box-threshold 0.35`

Primary image for pyjama whole body:
`python test.py --prompt "whole human body." --input ./input/pyjama-wholebody --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_pyjama-wholebody3 --mode primary --place center --pad-x 0.0 --pad-y 0.01 --box-threshold 0.0`

## Results

![`bottom leggins pants.` prompt](out_leggins/95031_DCM_M1+_dino.jpg){height=640}
![`upper underware bra.` prompt](out_panties/07010_MFE_W1+_dino.jpg){height=640}
