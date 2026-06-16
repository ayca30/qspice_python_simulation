from QSPICEBatchRunner import QSPICEBatchRunner
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# SETUP
# -----------------------------
runner = QSPICEBatchRunner(
    basefile="Controller_Test",
    workdir="control_batch_ilc"
)

runner.qsch_to_cir(
    r"C:\Users\aycah\Documents\RISE\qspice_and_training\Controller_Test.qsch"
)

# -----------------------------
# ILC SETTINGS
# -----------------------------
TARGET_PEAK = 1.0
f = 100
learning_rate = 0.5  # simple P-type ILC gain

# start with a guess for VAMP
vamp = 1.0

# -----------------------------
# LOGGING
# -----------------------------
vamp_history = []
error_history = []
current_history = []
waveform_history = []

# -----------------------------
# ILC LOOP
# -----------------------------
for k in range(20):

    cir_files = runner.generate_param_cir_files([
        {"VAMP": vamp}
    ])

    result = runner.run_batch(
        cir_files,
        signals=["I(L1)"],
        max_workers=1
    )[0]

    df = result["data"]
    t = df["Time"].to_numpy()
    i = df["I(L1)"].to_numpy()

    waveform_history.append((t.copy(), i.copy()))

    # measure steady-state peak (last cycle only)
    T = 1.0 / f
    last_cycle_mask = t >= (t[-1] - T)
    i_peak = np.max(np.abs(i[last_cycle_mask]))

    # ILC update — simple proportional correction
    error = TARGET_PEAK - i_peak
    vamp = vamp + learning_rate * error

    vamp_history.append(vamp)
    error_history.append(error)
    current_history.append(i_peak)

    print(f"iter {k:02d} | VAMP={vamp:.4f} | I_peak={i_peak:.4f} | error={error:.4f}")

# -----------------------------
# PLOTS
# -----------------------------

plt.figure()
plt.plot(vamp_history)
plt.title("VAMP convergence (ILC)")
plt.xlabel("Iteration")
plt.ylabel("VAMP")
plt.grid()

plt.figure()
plt.plot(error_history)
plt.axhline(0, linestyle="--", color='r')
plt.title("Peak current error (ILC)")
plt.xlabel("Iteration")
plt.ylabel("Error (A)")
plt.grid()

plt.figure()
plt.plot(current_history, marker='o')
plt.axhline(TARGET_PEAK, linestyle="--", color='r', label="Target")
plt.title("Peak current tracking (ILC)")
plt.xlabel("Iteration")
plt.ylabel("I_peak (A)")
plt.legend()
plt.grid()

plt.figure()
sim_duration = waveform_history[0][0][-1]
cmap = plt.cm.viridis(np.linspace(0, 1, len(waveform_history)))
for k, (t_k, i_k) in enumerate(waveform_history):
    plt.plot(t_k + k * sim_duration, i_k, color=cmap[k], alpha=0.8)
plt.axhline(TARGET_PEAK, color='r', linestyle='--', label="Target peak")
plt.axhline(-TARGET_PEAK, color='r', linestyle='--')
plt.title("Current vs Time (all iterations, ILC)")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis',
             norm=plt.Normalize(0, len(waveform_history)-1)),
             ax=plt.gca(), label="Iteration")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()