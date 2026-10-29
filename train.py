"""CreditPulse Top-Level Training Entry Point.
Run directly via:
    python train.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.train import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline()
