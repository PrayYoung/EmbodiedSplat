from pathlib import Path


def load_nerfstudio_pipeline(model_dir: str | Path):
    """
    Load a nerfstudio pipeline from a trained splatfacto model.

    model_dir should contain:
        config.yml
        nerfstudio_models/

    Returns:
        pipeline object
    """
    from nerfstudio.utils.eval_utils import eval_setup

    config_path = Path(model_dir) / "config.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yml not found in {model_dir}")

    _, pipeline, _, _ = eval_setup(
        config_path,
        test_mode="inference",
    )

    return pipeline
