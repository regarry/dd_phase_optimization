import os
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

from training.training import DynamicSpatialWellLoss

# ==========================================
# 2. Plotting Function (No Recalculation)
# ==========================================
def plot_existing_curve(loss_module):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"spatial_loss_curve_{timestamp}.png"
    if loss_module.last_cost_map is None:
        raise ValueError("The loss module hasn't processed any data yet. Run a forward pass first!")

    # Extract the exact array calculated inside the PyTorch forward pass
    y_values = loss_module.last_cost_map.numpy()
    
    # Reconstruct the symmetric X-axis (Minus to Plus) based solely on its layout width
    num_cols = len(y_values)
    center = (num_cols - 1) / 2
    x_values = torch.arange(num_cols).float().numpy() - center
    
    # Matplotlib Graph Setup
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, color='#6C5CE7', linewidth=2.5, label='Internal Cost Map Output')
    
    # Dynamically draw configuration milestones based on loaded YAML variables
    plt.axvline(x=loss_module.x, color='#2ECC71', linestyle='--', alpha=0.8, label=f'x_bound ({loss_module.x})')
    plt.axvline(x=-loss_module.x, color='#2ECC71', linestyle='--')
    plt.axvline(x=loss_module.y, color='#E74C3C', linestyle=':', alpha=0.8, label=f'y_target ({loss_module.y})')
    plt.axvline(x=-loss_module.y, color='#E74C3C', linestyle=':')
    
    # Fill background zones for context mapping
    plt.axvspan(-loss_module.x, loss_module.x, color='#2ECC71', alpha=0.08, label='Zero Penalty Well')
    
    plt.title('DynamicSpatialWellLoss: Extracted Cost Map Profile', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(r'Distance from Grid Center', fontsize=11)
    plt.ylabel('Cost Value', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # --- CHANGED: Save the file to disk instead of blocking with plt.show() ---
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Clear memory
    print(f"Success! Plot successfully saved to: {os.path.abspath(filename)}")


# ==========================================
# 3. Main Configuration & Execution Workflow
# ==========================================
if __name__ == "__main__":
    config_path = "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find '{config_path}'. Please create it in this directory.")
        
    # Load configuration parameters
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        

    #print(f"Loaded configuration details: {config}")

    # Initialize the loss class with values directly parsed from the YAML structure
    spatial_loss_fn = DynamicSpatialWellLoss(
        x_bound=config["spatial_well_x"], 
        y_target=config["spatial_well_y"]
    )
    
    # Generate mock tensor matching the loaded width config to trigger a forward cycle
    # Format: (Batch size=1, Rows/Channels=10, Columns=num_cols)
    mock_grid_width = 150 #config["N"]
    mock_input_data = torch.rand( mock_grid_width, mock_grid_width)
    
    print("Executing forward pass through the network model...")
    _ = spatial_loss_fn(mock_input_data)
    
    print("Extracting internal tensors and rendering graph...")
    plot_existing_curve(spatial_loss_fn)