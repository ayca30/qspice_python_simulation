from QSPICEBatchRunner import QSPICEBatchRunner
import numpy as np
import os
from scipy.interpolate import interp1d

# =============================================================
# SETUP
# =============================================================

runner = QSPICEBatchRunner(
    basefile="Controller_Test_Diverse_Training",
    workdir="CHECK_fft_ilc_single_case"
)

runner.qsch_to_cir(
    r"C:\Users\aycah\Documents\RISE\qspice_and_training\NEURAL_NETWORK_TRAINING\from VM\Controller_Test_Diverse_Training.qsch"
)

# =============================================================
# EXACT PHASE 4 TEST CASE
# =============================================================

f = 150
V_amp = 0.75
I_target = 1.0

LVAL = 1e-3
ISAT = 3
LSAT = 10e-6
RVAL = 1.0

dt = 1e-6
n_harmonics = 12

rms_target = 0.075
max_iterations = 200
max_consecutive_fails = 20
rms_improvement_tol = 1e-6

alpha = 0.4
alpha_min = 0.005
alpha_max = 0.9
alpha_fail_factor = 0.9
alpha_recovery_factor = 1.05

V_CLIP = 10

# =============================================================
# SIGNALS
# =============================================================

t = np.arange(0, 1 / f, dt)
freqs = np.fft.rfftfreq(len(t), d=dt)

i_reference = I_target * np.sin(2 * np.pi * f * t)
voltage = V_amp * np.sin(2 * np.pi * f * t)

# =============================================================
# HELPERS
# =============================================================

def calculate_thd(signal, f, dt):
    N = len(signal)
    freqs_ = np.fft.rfftfreq(N, d=dt)
    spectrum = np.abs(np.fft.rfft(signal)) / N

    f1_idx = np.argmin(np.abs(freqs_ - f))
    V1 = spectrum[f1_idx]

    if V1 == 0:
        return np.inf

    harmonics = [
        spectrum[np.argmin(np.abs(freqs_ - h * f))]
        for h in range(2, 10)
    ]

    return np.sqrt(sum(v**2 for v in harmonics)) / V1 * 100


def run_qspice(voltage):
    voltage_file = os.path.join(runner.workdir, "voltage.txt")
    np.savetxt(voltage_file, np.column_stack((t, voltage)))

    param_list = [{
        "LVAL": LVAL,
        "ISATVAL": ISAT,
        "LSATVAL": LSAT,
        "RVAL": RVAL,
        "TSTOP": t[-1],
        "TSTEP": t[1] - t[0],
    }]

    cir_files = runner.generate_param_cir_files(param_list)

    result = runner.run_batch(
        cir_files,
        signals=["I(L1)"],
        max_workers=1
    )[0]

    df = result["data"]

    time_sim = df.iloc[:, 0].to_numpy()
    current = df["I(L1)"].to_numpy()

    current = interp1d(
        time_sim,
        current,
        kind="linear",
        fill_value="extrapolate"
    )(t)

    return current


def fft_ilc_update(voltage_best, error_best, alpha):
    E_fft = np.fft.rfft(error_best)

    V_fft = np.fft.rfft(voltage_best)
    V_fft_new = V_fft.copy()

    for h in range(1, n_harmonics + 1):
        idx = np.argmin(np.abs(freqs - h * f))
        V_fft_new[idx] += alpha * E_fft[idx]

    voltage_new = np.fft.irfft(V_fft_new, n=len(t))
    voltage_new = np.clip(voltage_new, -V_CLIP, V_CLIP)

    return voltage_new


# =============================================================
# FFT-ILC LOOP
# =============================================================

rms_best = np.inf
voltage_best = voltage.copy()
error_best = np.zeros(len(t))
consecutive_fails = 0

rms_history = []
thd_history = []
alpha_history = []

print("\n" + "=" * 70)
print("CHECKING WHETHER EXACT PHASE 4 CASE CONVERGES WITH FFT-ILC")
print("=" * 70)
print(f"f={f} Hz | V_amp={V_amp} | L={LVAL} H | ISAT={ISAT} | LSAT={LSAT}")
print(f"Target RMS <= {rms_target}")
print(f"Voltage clip: ±{V_CLIP} V")
print("=" * 70)

for k in range(1, max_iterations + 1):

    current = run_qspice(voltage)

    error = i_reference - current
    rms = np.sqrt(np.mean(error**2))
    thd = calculate_thd(current, f, dt)

    rms_history.append(rms)
    thd_history.append(thd)
    alpha_history.append(alpha)

    print(
        f"Iter {k:04d} | "
        f"RMS {rms:.6f} A | "
        f"THD {thd:.3f}% | "
        f"alpha {alpha:.5f} | "
        f"Vpk {np.max(np.abs(voltage)):.3f} V"
    )

    if rms <= rms_target:
        rms_best = rms
        voltage_best = voltage.copy()
        error_best = error.copy()

        print("\nSUCCESS: FFT-ILC reached the RMS target.")
        break

    if rms < rms_best - rms_improvement_tol:
        rms_best = rms
        voltage_best = voltage.copy()
        error_best = error.copy()
        consecutive_fails = 0

        old_alpha = alpha
        alpha = min(alpha_max, alpha * alpha_recovery_factor)

        print(
            f"  Accepted | best RMS {rms_best:.6f} | "
            f"alpha {old_alpha:.5f} -> {alpha:.5f}"
        )

    else:
        consecutive_fails += 1

        old_alpha = alpha
        alpha = max(alpha_min, alpha * alpha_fail_factor)

        print(
            f"  Rejected | best RMS {rms_best:.6f} | "
            f"alpha {old_alpha:.5f} -> {alpha:.5f} | "
            f"fails {consecutive_fails}/{max_consecutive_fails}"
        )

        voltage = voltage_best.copy()

        if consecutive_fails >= max_consecutive_fails:
            print("\nSTOPPED: Too many consecutive failed updates.")
            break

    voltage = fft_ilc_update(voltage_best, error_best, alpha)

else:
    print("\nSTOPPED: Max iterations reached.")

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
print(f"Best RMS reached : {rms_best:.6f} A")
print(f"Final RMS        : {rms_history[-1]:.6f} A")
print(f"Final THD        : {thd_history[-1]:.3f}%")
print(f"Final alpha      : {alpha:.6f}")
print(f"Best voltage min : {voltage_best.min():.3f} V")
print(f"Best voltage max : {voltage_best.max():.3f} V")
print(f"Best voltage pk  : {np.max(np.abs(voltage_best)):.3f} V")

best_overall_rms = min(rms_history)

comparison_target = 0.075

for i, rms in enumerate(rms_history, start=1):
    if rms <= comparison_target:
        print(
            f"FFT-ILC reached RMS <= {comparison_target:.3f} "
            f"at iteration {i}"
        )
        break
else:
    print(
        f"FFT-ILC never reached RMS <= {comparison_target:.2f}"
    )