import os
import numpy as np
import matplotlib.pyplot as plt

# Directories for output
DATA_DIR = "sweep_results/data"
PLOT_DIR = "sweep_results/plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

class SystemParams:
    def __init__(self, name, base_g, base_kappa, base_gamma, unit):
        self.name = name
        self.base_g = base_g
        self.base_kappa = base_kappa
        self.base_gamma = base_gamma
        self.unit = unit

# The 6 realistic systems identified from literature
systems = [
    SystemParams("Circuit_QED", base_g=100.0, base_kappa=1.0, base_gamma=0.1, unit="MHz"),
    SystemParams("Neutral_Atoms", base_g=10.0, base_kappa=1.0, base_gamma=3.0, unit="MHz"),
    SystemParams("Trapped_Ions", base_g=5.0, base_kappa=1.0, base_gamma=20.0, unit="MHz"),
    SystemParams("NV_Centers", base_g=5.0, base_kappa=10.0, base_gamma=0.5, unit="MHz"),
    SystemParams("InAs_Quantum_Dots", base_g=20.0, base_kappa=50.0, base_gamma=2.0, unit="GHz"),
    SystemParams("Plasmonic_Molecules", base_g=10.0, base_kappa=50.0, base_gamma=5.0, unit="THz")
]

def calc_lifetime_arr(g, kappa, gamma):
    """Vectorized calculation of polariton lifetime."""
    discriminant = ((kappa - gamma) / 2)**2 - 4 * g**2
    weak_gamma_eff = (kappa + gamma) / 2 - np.sqrt(np.maximum(discriminant, 0))
    strong_gamma_eff = (kappa + gamma) / 2
    gamma_eff = np.where(discriminant >= 0, weak_gamma_eff, strong_gamma_eff)
    return np.where(gamma_eff > 0, 1.0 / gamma_eff, np.nan)

def plot_phase_diagram(X, Y, Z, R_boundary, xlabel, ylabel, title, save_path):
    """Plots a 2D colormap (phase diagram) overlaying the exceptional point."""
    plt.figure(figsize=(8, 6))
    
    # Plot the lifetime as a colormap
    # Using log scale for color because lifetime varies over orders of magnitude
    cmap = plt.cm.magma
    pcm = plt.pcolormesh(X, Y, np.log10(Z), cmap=cmap, shading='auto')
    cbar = plt.colorbar(pcm)
    cbar.set_label('Log10(Lifetime)')
    
    # Overlay the Exceptional Point Boundary where R = 0
    # R = (kappa - gamma)^2 - 16*g^2
    plt.contour(X, Y, R_boundary, levels=[0], colors='white', linestyles='dashed', linewidths=2)
    
    # Add labels
    plt.text(0.05, 0.95, 'Weak Coupling', transform=plt.gca().transAxes, color='white', 
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))
    plt.text(0.95, 0.05, 'Strong Coupling', transform=plt.gca().transAxes, color='white', 
             fontsize=12, horizontalalignment='right', verticalalignment='bottom', bbox=dict(facecolor='black', alpha=0.5))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def run_experiment_A():
    """Experiment A: Fixed Gamma (Two-Level System). Sweep g and kappa."""
    print("Running Experiment A (Fixed Gamma)...")
    N = 400
    for sys in systems:
        # Sweep ranges (0.01x to 10x base values)
        g_vals = np.linspace(sys.base_g * 0.01, sys.base_g * 10, N)
        k_vals = np.linspace(sys.base_kappa * 0.01, sys.base_kappa * 10, N)
        G, K = np.meshgrid(g_vals, k_vals)
        gamma = sys.base_gamma
        
        # Calculate Lifetime
        lifetimes = calc_lifetime_arr(G, K, gamma)
        # Calculate Discriminant for Boundary
        R = (K - gamma)**2 - 16 * G**2
        
        # Save Data
        np.save(os.path.join(DATA_DIR, f"ExpA_{sys.name}_lifetimes.npy"), lifetimes)
        
        # Plot Phase Diagram
        title = fr"Exp A: {sys.name.replace('_', ' ')} Phase Diagram ($\gamma = {gamma}$ {sys.unit})"
        save_path = os.path.join(PLOT_DIR, f"ExpA_{sys.name}.png")
        plot_phase_diagram(G, K, lifetimes, R, fr"Coupling $g$ ({sys.unit})", fr"Cavity Decay $\kappa$ ({sys.unit})", title, save_path)

def run_experiment_B():
    """Experiment B: Fixed Kappa (Cavity). Sweep g and gamma."""
    print("Running Experiment B (Fixed Kappa)...")
    N = 400
    for sys in systems:
        g_vals = np.linspace(sys.base_g * 0.01, sys.base_g * 10, N)
        gamma_vals = np.linspace(sys.base_gamma * 0.01, sys.base_gamma * 10, N)
        G, Gamma = np.meshgrid(g_vals, gamma_vals)
        kappa = sys.base_kappa
        
        lifetimes = calc_lifetime_arr(G, kappa, Gamma)
        R = (kappa - Gamma)**2 - 16 * G**2
        
        np.save(os.path.join(DATA_DIR, f"ExpB_{sys.name}_lifetimes.npy"), lifetimes)
        
        title = fr"Exp B: {sys.name.replace('_', ' ')} Phase Diagram ($\kappa = {kappa}$ {sys.unit})"
        save_path = os.path.join(PLOT_DIR, f"ExpB_{sys.name}.png")
        plot_phase_diagram(G, Gamma, lifetimes, R, fr"Coupling $g$ ({sys.unit})", fr"Dot Decay $\gamma$ ({sys.unit})", title, save_path)

def run_experiment_C():
    """Experiment C: Fixed g (Coupling/Volume). Sweep kappa and gamma."""
    print("Running Experiment C (Fixed g)...")
    N = 400
    for sys in systems:
        k_vals = np.linspace(sys.base_kappa * 0.01, sys.base_kappa * 10, N)
        gamma_vals = np.linspace(sys.base_gamma * 0.01, sys.base_gamma * 10, N)
        K, Gamma = np.meshgrid(k_vals, gamma_vals)
        g = sys.base_g
        
        lifetimes = calc_lifetime_arr(g, K, Gamma)
        R = (K - Gamma)**2 - 16 * g**2
        
        np.save(os.path.join(DATA_DIR, f"ExpC_{sys.name}_lifetimes.npy"), lifetimes)
        
        title = fr"Exp C: {sys.name.replace('_', ' ')} Phase Diagram ($g = {g}$ {sys.unit})"
        save_path = os.path.join(PLOT_DIR, f"ExpC_{sys.name}.png")
        # In this plot, strong coupling is when R < 0, meaning 16g^2 > (k - gamma)^2
        # So we just overlay R
        
        # We need a custom plot_phase_diagram here because the axes are different (weak/strong regions change)
        plt.figure(figsize=(8, 6))
        cmap = plt.cm.magma
        pcm = plt.pcolormesh(K, Gamma, np.log10(lifetimes), cmap=cmap, shading='auto')
        cbar = plt.colorbar(pcm)
        cbar.set_label('Log10(Lifetime)')
        
        plt.contour(K, Gamma, R, levels=[0], colors='white', linestyles='dashed', linewidths=2)
        
        plt.xlabel(fr"Cavity Decay $\kappa$ ({sys.unit})")
        plt.ylabel(fr"Dot Decay $\gamma$ ({sys.unit})")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

if __name__ == "__main__":
    print("Starting Exhaustive Cavity QED Parameter Sweep...")
    run_experiment_A()
    run_experiment_B()
    run_experiment_C()
    print(f"Sweep complete! Data saved to {DATA_DIR} and plots to {PLOT_DIR}.")
