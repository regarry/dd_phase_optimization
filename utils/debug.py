import torch
import sys

class MemorySnapshot:
    """
    Context manager that records GPU memory history and dumps a snapshot
    if an error occurs.
    """
    def __init__(self, filename="memory_snapshot.pickle", max_entries=100000):
        self.filename = filename
        self.max_entries = max_entries

    def __enter__(self):
        try:
            torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
            print(f"🔴 Memory recording started. Snapshot will save to '{self.filename}' on crash.")
        except AttributeError:
            print("⚠️ Warning: PyTorch version too old for memory recording.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
            if exc_type:
                print(f"\n💥 Crash detected: {exc_val}")
                print(f"💾 Dumping memory snapshot to {self.filename}...")
                torch.cuda.memory._dump_snapshot(self.filename)
                print("✅ Snapshot saved successfully.")
        except AttributeError:
            pass

def print_all_gpu_stats():
    """Prints memory usage for all available GPUs."""
    if not torch.cuda.is_available():
        return
        
    num_devices = torch.cuda.device_count()
    print("-" * 30)
    for i in range(num_devices):
        prop = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        print(f"GPU {i}: {prop.name} | Alloc: {allocated:.2f} MB | Res: {reserved:.2f} MB | Cap: {prop.total_memory / 1024**2:.2f} MB")
    print("-" * 30)