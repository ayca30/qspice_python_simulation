from QSPICEBatchRunner import QSPICEBatchRunner
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import os
from scipy.interpolate import interp1d

# =============================================================
# USER SETTINGS
# =============================================================

DATASET_FILE = "multi_ilc_dataset_final_target_12h.pkl"
MODEL_FILE = "best_final_target_model.pth"

QSPICE_FILE = r"C:\Users\aycah\Documents\RISE\qspice_and_training\NEURAL_NETWORK_TRAINING\from VM\Controller_Test_Diverse_Training.qsch"

WORKDIR = "COMPARE_NN_vs_FFT_ILC"
BASEFILE = "Controller_Test_Diverse_Training"

# Same exact test case
f = 150
V_amp = 0.75
I_target = 1.0

LVAL = 1e-3
ISAT = 1.0
LSAT = 5e-6
RVAL = 1.0

dt = 1e-6
n_harmonics = 12

rms_target = 0.06
max_iterations = 50

alpha_init = 0.4
alpha_min = 0.005
alpha_max = 0.9
alpha_fail_factor = 0.9
alpha_recovery_factor = 1.05

V_CLIP = 10
rms_improvement_tol = 1e-6
max_consecutive_fails = max_iterations

NN_SCALE = 1.0

# =============================================================
# LOAD DATASET NORMALIZATION
# =============================================================

with open(DATASET_FILE, "rb") as f_in:
    data = pickle.load(f_in)

X = data["X"]
Y = data["Y"]

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
Y_mean = Y.mean(axis=0)
Y_std = Y.std(axis=0) + 1e-8

# =============================================================
# TIME SIGNALS
# =============================================================

t = np.arange(0, 1 / f, dt)
freqs = np.fft.rfftfreq(len(t), d=dt)

i_reference = I_target * np.sin(2 * np.pi * f * t)
v_initial = V_amp * np.sin(2 * np.pi * f * t)

# =============================================================
# QSPICE SETUP
# =============================================================

runner = QSPICEBatchRunner(
    basefile=BASEFILE,
    workdir=WORKDIR
)

runner.qsch_to_cir(QSPICE_FILE)

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


def extract_harmonic_features(signal, f, freqs, n_harmonics):
    spectrum = np.fft.rfft(signal)
    features = []

    for h in range(1, n_harmonics + 1):
        idx = np.argmin(np.abs(freqs - h * f))
        features.append(spectrum[idx].real)
        features.append(spectrum[idx].imag)

    return np.array(features, dtype=np.float32)


def spectrum_to_time(harmonic_features, freqs, f, n_harmonics, n_time):
    spectrum = np.zeros(len(freqs), dtype=complex)

    for h in range(n_harmonics):
        idx = np.argmin(np.abs(freqs - (h + 1) * f))
        spectrum[idx] = harmonic_features[2*h] + 1j * harmonic_features[2*h + 1]

    return np.fft.irfft(spectrum, n=n_time)


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


# =============================================================
# NN MODEL
# =============================================================

class ILC_Net(nn.Module):
    def __init__(self, n_input_features, n_output_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_output_features)
        )

    def forward(self, x):
        return self.net(x)


model = ILC_Net(X.shape[1], Y.shape[1])
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()

# =============================================================
# FFT-ILC UPDATE
# =============================================================

def fft_ilc_update(voltage_best, error_best, alpha):
    E_fft = np.fft.rfft(error_best)
    V_fft = np.fft.rfft(voltage_best)

    V_fft_new = V_fft.copy()

    for h in range(1, n_harmonics + 1):
        idx = np.argmin(np.abs(freqs - h * f))
        V_fft_new[idx] += alpha * E_fft[idx]

    voltage_new = np.fft.irfft(V_fft_new, n=len(t))
    return np.clip(voltage_new, -V_CLIP, V_CLIP)


# =============================================================
# NN UPDATE
# =============================================================

def build_input_vector(
    error_signal,
    voltage_signal,
    scale_feature,
    f,
    V_amp,
    I_target,
    LVAL,
    ISAT,
    LSAT
):
    error_features = extract_harmonic_features(
        error_signal, f, freqs, n_harmonics
    )

    voltage_features = extract_harmonic_features(
        voltage_signal, f, freqs, n_harmonics
    )

    scale_feature_array = np.array([scale_feature], dtype=np.float32)

    plant_features = np.array([
        f / 1000,
        V_amp / 4,
        I_target / 2.5,
        LVAL / 0.01,
        ISAT / 10,
        LSAT / 50e-6
    ], dtype=np.float32)

    return np.concatenate([
        error_features,
        voltage_features,
        scale_feature_array,
        plant_features
    ]).astype(np.float32)

def nn_final_target_voltage_update(
    error_signal,
    voltage_current,
    model,
    f,
    freqs,
    n_harmonics,
    X_mean,
    X_std,
    Y_mean,
    Y_std,
    t,
    scale_feature,
    V_amp,
    I_target,
    LVAL,
    ISAT,
    LSAT,
    apply_scale=1.0,
    verbose=True
):

    model.eval()

    with torch.no_grad():

        x = build_input_vector(
            error_signal=error_signal,
            voltage_signal=voltage_current,
            scale_feature=scale_feature,
            f=f,
            V_amp=V_amp,
            I_target=I_target,
            LVAL=LVAL,
            ISAT=ISAT,
            LSAT=LSAT
        )

        x_norm_np = (x - X_mean) / X_std
        x_norm = torch.tensor(x_norm_np, dtype=torch.float32)

        y_norm = model(x_norm).numpy()
        y = y_norm * Y_std + Y_mean

    correction = spectrum_to_time(
        y,
        freqs,
        f,
        n_harmonics,
        len(t)
    )

    voltage_new = voltage_current + apply_scale * correction
    voltage_new = np.clip(voltage_new, -V_CLIP, V_CLIP)

    if verbose:
        print("\nNN FINAL-TARGET PREDICTION DIAGNOSTIC")
        print(f"  Input normalized range       : {x_norm_np.min():.3f} to {x_norm_np.max():.3f}")
        print(f"  Predicted coeff range        : {y.min():.3f} to {y.max():.3f}")
        print(f"  Training Y coeff range       : {Y.min():.3f} to {Y.max():.3f}")
        print(f"  Correction peak              : {np.max(np.abs(correction)):.4f} V")
        print(f"  Correction RMS               : {np.sqrt(np.mean(correction**2)):.4f} V")
        print(f"  Apply scale                  : {apply_scale:.3f}")
        print(f"  Voltage new peak             : {np.max(np.abs(voltage_new)):.4f} V")

        clipped_fraction = np.mean(np.abs(voltage_current + apply_scale * correction) > V_CLIP)
        print(f"  Clipped sample fraction      : {100 * clipped_fraction:.2f}%")

    return voltage_new, correction, y



# =============================================================
# GENERIC CONTROLLER LOOP
# =============================================================

def run_controller(controller_name):
    voltage = v_initial.copy()
    voltage_best = voltage.copy()
    error_best = np.zeros(len(t))

    rms_best = np.inf
    alpha = alpha_init
    consecutive_fails = 0

    rms_history = []
    thd_history = []
    current_history = []
    voltage_history = []

    reached_iter = None

    print("\n" + "=" * 70)
    print(f"RUNNING {controller_name}")
    print("=" * 70)

    for k in range(1, max_iterations + 1):
        current = run_qspice(voltage)
        error = i_reference - current

        rms = np.sqrt(np.mean(error**2))
        thd = calculate_thd(current, f, dt)

        rms_history.append(rms)
        thd_history.append(thd)
        current_history.append(current.copy())
        voltage_history.append(voltage.copy())

        print(
            f"{controller_name} Iter {k:04d} | "
            f"RMS {rms:.6f} A | "
            f"THD {thd:.3f}% | "
            f"alpha {alpha:.5f} | "
            f"Vpk {np.max(np.abs(voltage)):.3f} V"
        )

        if rms <= rms_target:
            reached_iter = k
            rms_best = rms
            voltage_best = voltage.copy()
            error_best = error.copy()
            print(f"\n{controller_name} SUCCESS at iteration {k}")
            break

        if rms < rms_best - rms_improvement_tol:
            rms_best = rms
            voltage_best = voltage.copy()
            error_best = error.copy()
            consecutive_fails = 0
            alpha = min(alpha_max, alpha * alpha_recovery_factor)
        else:
            consecutive_fails += 1
            alpha = max(alpha_min, alpha * alpha_fail_factor)
            voltage = voltage_best.copy()

            if consecutive_fails >= max_consecutive_fails:
                print(f"\n{controller_name} stopped: too many failed updates.")
                break

        if controller_name == "FFT-ILC":
            voltage = fft_ilc_update(voltage_best, error_best, alpha)

        elif controller_name == "NN":

            candidate_scales = [0.1, 0.3, 0.5, 1
            ]

            best_candidate = None

            for test_scale in candidate_scales:

                voltage_candidate, correction_candidate, y_candidate = nn_final_target_voltage_update(
                    error_signal=error_best,
                    voltage_current=voltage_best,
                    model=model,
                    f=f,
                    freqs=freqs,
                    n_harmonics=n_harmonics,
                    X_mean=X_mean,
                    X_std=X_std,
                    Y_mean=Y_mean,
                    Y_std=Y_std,
                    t=t,
                    scale_feature=alpha,
                    V_amp=V_amp,
                    I_target=I_target,
                    LVAL=LVAL,
                    ISAT=ISAT,
                    LSAT=LSAT,
                    apply_scale=test_scale,
                    verbose=False
                )

                current_candidate = run_qspice(voltage_candidate)
                error_candidate = i_reference - current_candidate
                rms_candidate = np.sqrt(np.mean(error_candidate**2))
                thd_candidate = calculate_thd(current_candidate, f, dt)

                if best_candidate is None or rms_candidate < best_candidate["rms"]:
                    best_candidate = {
                        "scale": test_scale,
                        "voltage": voltage_candidate,
                        "rms": rms_candidate,
                        "thd": thd_candidate
                    }

            print(
                f"  NN chosen scale {best_candidate['scale']:.2f} | "
                f"candidate RMS {best_candidate['rms']:.6f} A | "
                f"candidate THD {best_candidate['thd']:.3f}%"
            )

            voltage = best_candidate["voltage"]

        else:
            raise ValueError("Unknown controller name.")

    return {
        "rms_history": rms_history,
        "thd_history": thd_history,
        "current_history": current_history,
        "voltage_history": voltage_history,
        "reached_iter": reached_iter,
        "best_rms": min(rms_history),
        "best_idx": int(np.argmin(rms_history)),
    }


# =============================================================
# RUN BOTH CONTROLLERS
# =============================================================

fft_results = run_controller("FFT-ILC")
nn_results = run_controller("NN")

# =============================================================
# CHOOSE FAIR COMPARISON ITERATION
#
# Case 1: both converge
#   -> show both at the earlier convergence iteration
#
# Case 2: one converges, one does not
#   -> show both at the convergence iteration
#
# Case 3: neither converges
#   -> show both at the final/max iteration
# =============================================================

convergence_iters = []

if fft_results["reached_iter"] is not None:
    convergence_iters.append(fft_results["reached_iter"])

if nn_results["reached_iter"] is not None:
    convergence_iters.append(nn_results["reached_iter"])

if len(convergence_iters) > 0:
    compare_iter = min(convergence_iters)
else:
    compare_iter = min(
        len(fft_results["rms_history"]),
        len(nn_results["rms_history"])
    )

fft_idx = min(compare_iter - 1, len(fft_results["current_history"]) - 1)
nn_idx = min(compare_iter - 1, len(nn_results["current_history"]) - 1)

fft_plot_iter = fft_idx + 1
nn_plot_iter = nn_idx + 1

fft_current_compare = fft_results["current_history"][fft_idx]
nn_current_compare = nn_results["current_history"][nn_idx]

fft_rms_compare = fft_results["rms_history"][fft_idx]
nn_rms_compare = nn_results["rms_history"][nn_idx]

fft_thd_compare = fft_results["thd_history"][fft_idx]
nn_thd_compare = nn_results["thd_history"][nn_idx]

# =============================================================
# PRINT SUMMARY
# =============================================================

print("\n" + "=" * 70)
print("SIDE-BY-SIDE RESULT")
print("=" * 70)

print(f"RMS target: {rms_target:.6f} A")

if nn_results["reached_iter"] is not None:
    print(f"NN reached target at iteration {nn_results['reached_iter']}")
else:
    print("NN did not reach target.")

if fft_results["reached_iter"] is not None:
    print(f"FFT-ILC reached target at iteration {fft_results['reached_iter']}")
else:
    print("FFT-ILC did not reach target.")

print(f"\nComparison iteration: {compare_iter}")
print(f"FFT-ILC RMS at comparison iteration: {fft_rms_compare:.6f} A")
print(f"NN RMS at comparison iteration     : {nn_rms_compare:.6f} A")

# =============================================================
# PLOTS
# =============================================================

t_ms = t * 1000

plt.figure(figsize=(14, 8))

# FFT waveform
plt.subplot(2, 2, 1)
plt.plot(t_ms, fft_current_compare, label=f"FFT-ILC current")
plt.plot(t_ms, i_reference, "--", label="Reference")
plt.xlabel("Time (ms)")
plt.ylabel("Current (A)")
plt.title(
    f"FFT-ILC at Iteration {fft_plot_iter}\n"
    f"RMS={fft_rms_compare:.4f} A, THD={fft_thd_compare:.2f}%"
)
plt.grid(True)
plt.legend()

# NN waveform
plt.subplot(2, 2, 2)
plt.plot(t_ms, nn_current_compare, label="NN current")
plt.plot(t_ms, i_reference, "--", label="Reference")
plt.xlabel("Time (ms)")
plt.ylabel("Current (A)")
plt.title(
    f"NN at Iteration {nn_plot_iter}\n"
    f"RMS={nn_rms_compare:.4f} A, THD={nn_thd_compare:.2f}%"
)
plt.grid(True)
plt.legend()

# RMS convergence
plt.subplot(2, 1, 2)
plt.plot(
    range(1, len(fft_results["rms_history"]) + 1),
    fft_results["rms_history"],
    label="FFT-ILC RMS"
)
plt.plot(
    range(1, len(nn_results["rms_history"]) + 1),
    nn_results["rms_history"],
    label="NN RMS"
)
plt.axhline(rms_target, linestyle="--", label=f"Target RMS = {rms_target}")
plt.axvline(compare_iter, linestyle=":", label=f"Comparison iteration = {compare_iter}")

plt.xlabel("Iteration")
plt.ylabel("RMS Error (A)")
plt.title("Convergence Comparison")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()