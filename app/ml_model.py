import joblib

from app.config import settings

THRESHOLD = 0.5

_dv = None
_model = None


def load_model() -> None:
    """Load the model from disk once, when the app starts."""
    global _dv, _model
    bundle = joblib.load(settings.model_path)
    _dv, _model = bundle["dv"], bundle["model"]
    n_dv = len(_dv.get_feature_names_out())
    if n_dv != _model.n_features_in_:
        raise RuntimeError(
            f"Bad model file: vectorizer has {n_dv} features, model expects {_model.n_features_in_}"
        )
    print(f"Model loaded ({n_dv} features)")


def predict_churn(features: dict) -> tuple[bool, float]:
    """Return (churn_prediction, churn_probability) for one customer."""
    X = _dv.transform([features])
    proba = float(_model.predict_proba(X)[0, 1])
    return proba >= THRESHOLD, round(proba, 4)
