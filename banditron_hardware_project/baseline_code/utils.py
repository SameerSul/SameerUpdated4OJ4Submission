import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curve(aer_history, save_path=None):
    """Plot average error rate over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(aer_history)
    plt.xlabel('Sample')
    plt.ylabel('Average Error Rate')
    plt.title('Banditron Online Learning Curve')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compare_experiments(results_list, metric='train_aer'):
    """Compare multiple experimental runs"""
    datasets = [r['dataset'] for r in results_list]
    values = [r[metric] for r in results_list]
    
    plt.figure(figsize=(12, 6))
    plt.bar(datasets, values)
    plt.xlabel('Dataset')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Comparison: {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def estimate_fpga_resources(n_features, n_classes):
    """Estimate FPGA resource requirements"""
    # Weight storage
    weight_bits = n_features * n_classes * 16  # 16-bit fixed point
    weight_bytes = weight_bits // 8
    
    # MAC units needed
    mac_units = n_features * n_classes
    
    # Estimate for Xilinx Virtex-6 (from Q-Learning paper)
    slice_luts = mac_units * 10  # Rough estimate
    slice_registers = mac_units * 8
    dsp48s = mac_units  # Each MAC uses 1 DSP48
    
    print(f"\nEstimated FPGA Resources:")
    print(f"  Weight storage: {weight_bytes:,} bytes ({weight_bytes/1024:.1f} KB)")
    print(f"  MAC operations/sample: {mac_units:,}")
    print(f"  Estimated Slice LUTs: {slice_luts:,}")
    print(f"  Estimated Slice Registers: {slice_registers:,}")
    print(f"  Estimated DSP48 blocks: {dsp48s:,}")
    
    return {
        'weight_bytes': weight_bytes,
        'mac_units': mac_units,
        'slice_luts': slice_luts,
        'slice_registers': slice_registers,
        'dsp48s': dsp48s
    }