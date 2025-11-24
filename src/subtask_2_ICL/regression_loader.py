#loads regression checkpoint from subtask 1 -> need to run first
#usually best_model.pt
import os
import torch

def _read_head_bytes(path, n=8):
    with open(path, "rb") as f:
        return f.read(n)

def load_checkpoint_safe(path):
    """
    Load a PyTorch checkpoint with robustness for different saved formats and PyTorch >=2.6.
    Returns the raw object returned by torch.load.
    Raises informative errors if the file is not a valid checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    size = os.path.getsize(path)
    if size == 0:
        raise RuntimeError(f"Checkpoint file {path} is empty (size 0).")

    head = _read_head_bytes(path, 4)
    # Quick heuristic: many binary pickles start with b'\x80\x04' (pickle), or ELF / other magic.
    # If the file looks text-like, show helpful message.
    text_like = False
    try:
        head.decode("utf-8")
        # If first bytes decode as text and first char is '{' or '\n', treat as text-like
        if head.strip().startswith(b"{") or head.strip().startswith(b"[") or head.strip().startswith(b"\n"):
            text_like = True
    except Exception:
        text_like = False

    if text_like:
        # still try loading with torch but warn
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            return ck
        except Exception as e:
            raise RuntimeError(
                f"File {path} appears text-like (maybe JSON/JSONL) and failed to load as a PyTorch checkpoint.\n"
                f"Please confirm you pointed --reg_checkpoint to the correct binary checkpoint file. Original error: {e}"
            ) from e

    # Try guarded torch.load with weights_only=False for PyTorch 2.6 compatibility
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return ck
    except TypeError:
        # Older torch may not support weights_only argument; try without it
        try:
            ck = torch.load(path, map_location="cpu")
            return ck
        except Exception as e2:
            raise RuntimeError(f"Failed to load checkpoint {path}: {e2}") from e2
    except Exception as e:
        # As a last resort, try with weights_only=True (safe but might not match)
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
            return ck
        except Exception:
            # Return the original error to user
            raise RuntimeError(
                f"Failed to load checkpoint {path} with multiple strategies. Last error: {e}"
            ) from e


def build_regressor_from_checkpoint(checkpoint_path, regressor_cls, regressor_kwargs=None, device="cpu"):
    """
    regressor_cls: class (not instance) of the regressor (e.g. TransformerVARegressor)
    regressor_kwargs: dict passed to constructor
    Returns: instantiated model (moved to device) with loaded weights.
    """
    regressor_kwargs = regressor_kwargs or {}
    ck = load_checkpoint_safe(checkpoint_path)

    # instantiate model
    model = regressor_cls(**regressor_kwargs)

    # ck can be:
    #  - a dict with 'model_state' or 'state_dict'
    #  - a pure state_dict
    #  - a full pickled model (rare)
    if isinstance(ck, dict):
        # common keys
        if "model_state" in ck:
            state = ck["model_state"]
            model.load_state_dict(state)
        elif "state_dict" in ck:
            state = ck["state_dict"]
            model.load_state_dict(state)
        else:
            # maybe the dict *is* a state_dict
            try:
                model.load_state_dict(ck)
            except Exception as e:
                # maybe ck is a pickled model (already an object)
                # If ck has keys like '__class__' it's not a state_dict; raise informative error
                raise RuntimeError(
                    "Checkpoint appears to be a generic dict but does not contain 'model_state' or 'state_dict'.\n"
                    "If you saved a full model object, try saving model.state_dict() instead.\n"
                    f"Original load error: {e}"
                ) from e
    else:
        # ck might be a pickled model - try to set state dict if possible
        # but in most stable workflows, ck should be dict/state_dict
        raise RuntimeError(
            "Checkpoint loaded but it is not a dict-like state_dict. Please ensure you saved the model with "
            "torch.save({'model_state': model.state_dict()}, PATH) or torch.save(model.state_dict(), PATH)."
        )

    model.to(device)
    model.eval()
    return model
