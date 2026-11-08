# Fly spam classifier (inference bundle)

Binary **spam / ham** email classifier. Email text is hashed onto a frozen FlyWire-derived olfactory subcircuit; spike counts feed a small Torch MLP.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Predict

```bash
python infer.py "URGENT: verify your account and claim your prize"
python infer.py --interactive
```

## Python API

```python
from infer import FlySpamClassifier
clf = FlySpamClassifier()  # this folder
print(clf.predict("Team standup notes for Monday"))
```

## Contents

| File | Purpose |
|---|---|
| `weights/spam_classifier.pt` | Trained Torch classifier |
| `weights/fly_engine.pkl` | Frozen 1200-neuron subcircuit used at inference |
| `infer.py` | CLI + `FlySpamClassifier` |
| `model_card.json` | Metadata |

You do **not** need the full ~852 MB FlyWire feather for inference.

## License

`fly_engine.pkl` is derived from the FlyWire connectome (**CC BY-NC 4.0**, non-commercial). Cite Dorkenwald et al. 2024 and Schlegel et al. 2024 if you publish with it.
