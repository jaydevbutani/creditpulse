"""CreditPulse Deep Learning Package."""
from model.model_config import (
    MODEL_NAME, MODEL_VERSION, INPUT_DIM, SEQUENCE_LENGTH, D_MODEL, N_LAYERS, N_HEADS
)

__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "INPUT_DIM",
    "SEQUENCE_LENGTH",
    "D_MODEL",
    "N_LAYERS",
    "N_HEADS",
    "MultiTaskDistressTransformer",
    "DistressScorer",
    "compute_integrated_gradients",
    "extract_top_drivers"
]

def __getattr__(name: str):
    if name == "MultiTaskDistressTransformer":
        from model.architecture import MultiTaskDistressTransformer
        return MultiTaskDistressTransformer
    if name == "DistressScorer":
        from model.inference import DistressScorer
        return DistressScorer
    if name in ("compute_integrated_gradients", "extract_top_drivers"):
        import model.explainability as exp
        return getattr(exp, name)
    raise AttributeError(f"module 'model' has no attribute '{name}'")
