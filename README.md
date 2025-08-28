# Groundingdino-test script (even for ARM Mac) as Fashion detector

Purpose of this repository is to somehow archive steps to run Grounding DINO object detection with text encoder.
Our use-case

## Setup (even for arch. ARM MAC)

Note: Make sure to run Python 3.9, 3.13 did not work as of 28/08/2025)

```
python3.9 -m venv venv install 3.11.9
source .venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.2,<2.4" torchvision torchaudio
pip install groundingdino-py psd-tools pillow opencv-pytho
curl -L -o weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

```

## Test commands:

`python test.py --prompt "bottom underware panties." --input ./input/bra --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_panties`

`python test.py --prompt "upper underware bra." --input ./input/bra --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu`

`python test.py --prompt "bottom leggins pants." --input ./input/leggins --weights weights/groundingdino_swint_ogc.pth --include-images --device cpu --out out_leggins`

## Results

![`bottom leggins pants.` prompt](out_leggins/95031_DCM_M1+_dino.jpg){height=640}
![`upper underware bra.` prompt](out_panties/07010_MFE_W1+_dino.jpg){height=640}
