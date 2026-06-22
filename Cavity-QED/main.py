import os
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, destroy, sigmam, sigmap, qeye, mesolve

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

class CavityQEDSystem:
    def __init__(self, name, g, kappa, gamma, time_max, g_range, kappa_range, gamma_range, unit):
        self.name = name
        self.g = g
        self.kappa = kappa
        self.gamma = gamma
        self.time_max = time_max
        self.g_range = g_range
        self.kappa_range = kappa_range
        self.gamma_range = gamma_range
        self.unit = unit

# Realistic parameter definitions for different Cavity QED systems

# 1. InAs Quantum Dots in Photonic Crystals (GHz)
# Usually operates in intermediate to strong coupling
inas_qd = CavityQEDSystem(
    name="InAs_Quantum_Dots",
    g=20.0, kappa=50.0, gamma=2.0, time_max=0.5,
    g_range=np.linspace(1, 50, 100),
    kappa_range=np.linspace(1, 100, 100),
    gamma_range=np.linspace(0.1, 10, 100),
    unit="GHz"
)

# 2. Circuit QED (Superconducting Qubits) (MHz)
# Very strong coupling regime
circuit_qed = CavityQEDSystem(
    name="Circuit_QED",
    g=100.0, kappa=1.0, gamma=0.1, time_max=5.0,
    g_range=np.linspace(10, 500, 100),
    kappa_range=np.linspace(0.01, 10, 100),
    gamma_range=np.linspace(0.01, 2, 100),
    unit="MHz"
)

# 3. Neutral Atoms in Fabry-Perot Cavities (MHz)
# Strong coupling possible but limited by atom transit time / position
neutral_atoms = CavityQEDSystem(
    name="Neutral_Atoms",
    g=10.0, kappa=1.0, gamma=3.0, time_max=2.0,
    g_range=np.linspace(1, 100, 100),
    kappa_range=np.linspace(0.1, 10, 100),
    gamma_range=np.linspace(1, 10, 100),
    unit="MHz"
)

# 4. Trapped Ions in Optical Cavities (MHz)
# Good localization, slightly different regime
trapped_ions = CavityQEDSystem(
    name="Trapped_Ions",
    g=5.0, kappa=1.0, gamma=20.0, time_max=2.0,
    g_range=np.linspace(1, 50, 100),
    kappa_range=np.linspace(0.1, 10, 100),
    gamma_range=np.linspace(1, 30, 100),
    unit="MHz"
)

# 5. NV Centers in Diamond (MHz)
# Phonon sidebands affect this, typically weak to intermediate coupling
nv_centers = CavityQEDSystem(
    name="NV_Centers",
    g=5.0, kappa=10.0, gamma=0.5, time_max=2.0,
    g_range=np.linspace(0.1, 50, 100),
    kappa_range=np.linspace(0.1, 50, 100),
    gamma_range=np.linspace(0.01, 2, 100),
    unit="MHz"
)

# 6. Plasmonic Molecules (THz)
# Huge g due to tiny mode volume, but massive kappa due to ohmic losses
plasmonic = CavityQEDSystem(
    name="Plasmonic_Molecules",
    g=10.0, kappa=50.0, gamma=5.0, time_max=1.0, # Values in THz
    g_range=np.linspace(1, 30, 100),
    kappa_range=np.linspace(10, 150, 100),
    gamma_range=np.linspace(1, 20, 100),
    unit="THz"
)

systems = [inas_qd, circuit_qed, neutral_atoms, trapped_ions, nv_centers, plasmonic]

def simulate_dynamics(system):
    """Simulates Jaynes-Cummings dynamics for a specific system."""
    N_cav = 2
    a = tensor(qeye(2), destroy(N_cav))
    sm = tensor(sigmam(), qeye(N_cav))
    sp = tensor(sigmap(), qeye(N_cav))

    H = system.g * (sp * a + sm * a.dag())
    c_ops = []
    if system.kappa > 0:
        c_ops.append(np.sqrt(system.kappa) * a)
    if system.gamma > 0:
        c_ops.append(np.sqrt(system.gamma) * sm)

    psi0 = tensor(basis(2, 0), basis(N_cav, 0))
    P_dot = sp * sm
    P_cav = a.dag() * a

    tlist = np.linspace(0, system.time_max, 500)
    result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[P_dot, P_cav])
    
    plt.figure(figsize=(8, 5))
    plt.plot(tlist, result.expect[0], 'b-', label='Atom/Dot $P_e$')
    plt.plot(tlist, result.expect[1], 'r--', label='Cavity $P_c$')
    plt.title(fr'{system.name.replace("_", " ")} Dynamics\n$g={system.g}, \kappa={system.kappa}, \gamma={system.gamma}$ {system.unit}')
    plt.xlabel(f'Time (1/{system.unit})')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{system.name}_dynamics.png")
    plt.close()

def calc_lifetime_arr(g, kappa, gamma):
    """Calculates polariton lifetimes vectorized for fast plotting."""
    discriminant = ((kappa - gamma) / 2)**2 - 4 * g**2
    weak_gamma_eff = (kappa + gamma) / 2 - np.sqrt(np.maximum(discriminant, 0))
    strong_gamma_eff = (kappa + gamma) / 2
    gamma_eff = np.where(discriminant >= 0, weak_gamma_eff, strong_gamma_eff)
    return np.where(gamma_eff > 0, 1.0 / gamma_eff, np.nan)

def plot_lifetimes(system):
    """Generates 3D and 2D lifetime plots for a specific system."""
    fig = plt.figure(figsize=(15, 10))

    # --- 3D Plot ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    K_3d, G_3d = np.meshgrid(system.kappa_range[::2], system.g_range[::2])
    T_3d = calc_lifetime_arr(G_3d, K_3d, system.gamma)
    surf = ax1.plot_surface(K_3d, G_3d, T_3d, cmap='magma', edgecolor='none', alpha=0.9)
    ax1.set_title(fr'3D Surface ($\gamma = {system.gamma}$ {system.unit})')
    ax1.set_xlabel(fr'$\kappa$ ({system.unit})')
    ax1.set_ylabel(fr'$g$ ({system.unit})')
    ax1.set_zlabel(f'Lifetime $\tau$')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # --- 2D Plot: vs kappa ---
    ax2 = fig.add_subplot(2, 2, 2)
    taus_k = calc_lifetime_arr(system.g, system.kappa_range, system.gamma)
    ax2.plot(system.kappa_range, taus_k, color='blue')
    ax2.set_title(fr'Lifetime vs $\kappa$ ($g={system.g}, \gamma={system.gamma}$ {system.unit})')
    ax2.set_xlabel(fr'Cavity Decay $\kappa$ ({system.unit})')
    ax2.set_ylabel('Lifetime $\tau$')
    ax2.grid(True, linestyle='--')

    # --- 2D Plot: vs gamma ---
    ax3 = fig.add_subplot(2, 2, 3)
    taus_gamma = calc_lifetime_arr(system.g, system.kappa, system.gamma_range)
    ax3.plot(system.gamma_range, taus_gamma, color='red')
    ax3.set_title(fr'Lifetime vs $\gamma$ ($g={system.g}, \kappa={system.kappa}$ {system.unit})')
    ax3.set_xlabel(fr'Dot Decay $\gamma$ ({system.unit})')
    ax3.set_ylabel('Lifetime $\tau$')
    ax3.grid(True, linestyle='--')

    # --- 2D Plot: vs g ---
    ax4 = fig.add_subplot(2, 2, 4)
    taus_g = calc_lifetime_arr(system.g_range, system.kappa, system.gamma)
    ax4.plot(system.g_range, taus_g, color='green')
    ax4.set_title(fr'Lifetime vs $g$ ($\kappa={system.kappa}, \gamma={system.gamma}$ {system.unit})')
    ax4.set_xlabel(fr'Coupling $g$ ({system.unit})')
    ax4.set_ylabel('Lifetime $\tau$')
    ax4.grid(True, linestyle='--')

    plt.suptitle(f'{system.name.replace("_", " ")} Lifetime Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"figures/{system.name}_lifetimes.png")
    plt.close()

if __name__ == "__main__":
    for sys in systems:
        print(f"Simulating {sys.name}...")
        simulate_dynamics(sys)
        plot_lifetimes(sys)
    print("Done! All figures saved in the 'figures/' directory.")
