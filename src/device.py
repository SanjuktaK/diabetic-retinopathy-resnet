import torch

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple MPS GPU")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")