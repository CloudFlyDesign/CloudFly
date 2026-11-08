# Fly face biometrics (Raspberry Pi bundle)

Low-res (**32×32**) face identification using a frozen flyputer subcircuit + Torch classifier.

## Recommended print target

**Camila Cabello** — highest holdout recall in this trained model (~53%).  
Print images are in `print_targets/Camila Cabello/`. Hold the printout to the Pi camera and tap **Identify face**.

## Setup on the Pi

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Then open `http://<pi-ip>:8787` on a phone/laptop on the same network (or locally on the Pi with a browser).

Allow camera permission. Prefer the rear camera for a printout demo.

## CLI

```bash
python infer.py path/to/face.jpg
```

## Notes

- No full FlyWire feather required — `weights/fly_engine.pkl` is enough.
- `fly_engine.pkl` is FlyWire-derived (**CC BY-NC**, non-commercial).
- Face photos in this demo come from the Kaggle face-recognition dataset (CC0).
