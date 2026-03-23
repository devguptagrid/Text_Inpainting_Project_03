import torch

def compute_model_size(model):
    total_params = sum(p.numel() for p in model.parameters()) ##Counts all weights in the model like embeddings, attention weights, FNN layers
    total_size_mb = total_params * 4 / (1024 ** 2) ##convert to memory -> float32 = 4 bytes per parameter, 1024^2=bytes->MB

    
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Model Size: {total_size_mb:.2f} MB")

    return total_size_mb
    

def get_mps_memory():
    try:
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    except:
        return 0.0

class MemoryTracker:
    def __init__(self): ## stores maximum memory seen so far
        self.peak = 0

    
    def update(self): ## at every step, check current memory and update max
        current = get_mps_memory()
        self.peak = max(self.peak, current)

    def get_peak(self): ## returns highest memory used during execution
        return self.peak
    
