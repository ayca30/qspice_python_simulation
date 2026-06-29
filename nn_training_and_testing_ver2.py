from QSPICEBatchRunner import QSPICEBatchRunner
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import pickle
import pandas as pd
from scipy.interpolate import interp1d

# =============================================================
# LOAD FINAL-TARGET DATASET
# =============================================================

TRAIN_DATASET_FILE = "multi_ilc_dataset_final_target_12h.pkl"
TEST_DATASET_FILE  = "multi_ilc_dataset_final_target_12h_independent.pkl"

USE_INDEPENDENT_TEST = False   # set True once you generate this file

USE_SAVED_MODEL = False
MODEL_FILE = "best_final_target_model.pth"

with open(TRAIN_DATASET_FILE, "rb") as f_in:
    data = pickle.load(f_in)

X = data["X"]
Y = data["Y"]

dt = data["dt"]
n_harmonics = data["n_harmonics"]
rms_target_dataset = data["rms_target"]

print("\nLoaded FINAL-TARGET ILC dataset.")
print(f"  Training samples : {X.shape[0]}")
print(f"  X shape          : {X.shape}")
print(f"  Y shape          : {Y.shape}")
print(f"  dt               : {dt}")
print(f"  n_harmonics      : {n_harmonics}")
print(f"  Dataset RMS target: {rms_target_dataset}")

if "accepted_cases" in data:
    print(f"  Accepted cases   : {data['accepted_cases']}")
if "rejected_cases" in data:
    print(f"  Rejected cases   : {data['rejected_cases']}")

# Safety checks
expected_x_features = 4 * n_harmonics + 7
expected_y_features = 2 * n_harmonics

if X.shape[1] != expected_x_features:
    raise ValueError(
        f"X feature mismatch. Expected {expected_x_features}, got {X.shape[1]}."
    )

if Y.shape[1] != expected_y_features:
    raise ValueError(
        f"Y feature mismatch. Expected {expected_y_features}, got {Y.shape[1]}."
    )

# =============================================================
# SETUP QSPICE
# =============================================================

runner = QSPICEBatchRunner(
    basefile="Controller_Test_Diverse_Training",
    workdir="NN_FINAL_TARGET_TEST"
)

runner.qsch_to_cir(
    r"C:\Users\aycah\Documents\RISE\qspice_and_training\NEURAL_NETWORK_TRAINING\from VM\Controller_Test_Diverse_Training.qsch"
)

# =============================================================
# PHASE 4 TEST CASE
# =============================================================

f = 150
V_amp = 0.75
I_target = 1.0

TEST_LVAL = 1e-3
TEST_ISAT = 3
TEST_LSAT = 10e-6
TEST_RVAL = 1.0

comparison_target = 0.05
max_iterations = 50
max_consecutive_fails = 10

V_CLIP = 10

# Since the NN now predicts a final-target correction,
# start lower than 1.0 for safety.
NN_SCALE_INIT = 0.50
NN_SCALE_MIN = 0.05
NN_SCALE_MAX = 1.00
NN_SCALE_ACCEPT_FACTOR = 1.10
NN_SCALE_FAIL_FACTOR = 0.70

t = np.arange(0, 1 / f, dt)
freqs = np.fft.rfftfreq(len(t), d=dt)
i_reference = I_target * np.sin(2 * np.pi * f * t)

# =============================================================
# HELPER FUNCTIONS
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
        for h in range(2, min(10, n_harmonics + 1))
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


def run_qspice(voltage, runner, t):
    voltage_file = os.path.join(runner.workdir, "voltage.txt")
    np.savetxt(voltage_file, np.column_stack((t, voltage)))

    param_list = [{
        "LVAL": TEST_LVAL,
        "ISATVAL": TEST_ISAT,
        "LSATVAL": TEST_LSAT,
        "RVAL": TEST_RVAL,
        "TSTOP": t[-1],
        "TSTEP": t[1] - t[0]
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


def rms_error(current):
    return np.sqrt(np.mean((i_reference - current) ** 2))


def safe_cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom < 1e-12:
        return 0.0

    return np.dot(a, b) / denom


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
        error_signal,
        f,
        freqs,
        n_harmonics
    )

    voltage_features = extract_harmonic_features(
        voltage_signal,
        f,
        freqs,
        n_harmonics
    )

    scale_feature_array = np.array(
        [scale_feature],
        dtype=np.float32
    )

    plant_features = np.array([
        f / 1000,
        V_amp / 4,
        I_target / 2.5,
        LVAL / 0.01,
        ISAT / 10,
        LSAT / 50e-6
    ], dtype=np.float32)

    x = np.concatenate([
        error_features,
        voltage_features,
        scale_feature_array,
        plant_features
    ])

    return x.astype(np.float32)


# =============================================================
# NORMALIZATION
# =============================================================

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8

Y_mean = Y.mean(axis=0)
Y_std = Y.std(axis=0) + 1e-8

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)

print("\nNormalization summary:")
print(f"  X range: {X.min():.3f} to {X.max():.3f}")
print(f"  Y range: {Y.min():.3f} to {Y.max():.3f}")
print(f"  X norm range: {X_norm.min():.3f} to {X_norm.max():.3f}")
print(f"  Y norm range: {Y_norm.min():.3f} to {Y_norm.max():.3f}")

# =============================================================
# MODEL
# =============================================================

class ILC_Net(nn.Module):

    def __init__(self, n_input_features, n_output_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_output_features)
        )

    def forward(self, x):
        return self.net(x)


n_input_features = X.shape[1]
n_output_features = Y.shape[1]

model = ILC_Net(
    n_input_features,
    n_output_features
)

print("\nModel:")
print(model)

# =============================================================
# TRAIN
# =============================================================

print("\n" + "=" * 60)
print("PHASE 3: TRAINING FINAL-TARGET NEURAL NETWORK")
print("=" * 60)

if USE_SAVED_MODEL:

    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()

    print(f"\nLoaded saved model: {MODEL_FILE}")

else:

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-6
    )

    loss_fn = nn.MSELoss()

    epochs = 1000
    train_losses = []

    for epoch in range(epochs):

        model.train()

        pred = model(X_tensor)
        loss = loss_fn(pred, Y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    print(f"\nTraining complete. Final loss: {train_losses[-1]:.6f}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =============================================================
# TRAINING DATASET EVALUATION
# =============================================================

model.eval()

with torch.no_grad():
    Y_pred_train_norm = model(X_tensor).numpy()

Y_pred_train = Y_pred_train_norm * Y_std + Y_mean

train_mse = np.mean((Y_pred_train - Y) ** 2)
train_rmse = np.sqrt(train_mse)

print("\n" + "=" * 60)
print("TRAINING DATASET CHECK")
print("=" * 60)
print(f"Training MSE  : {train_mse:.6e}")
print(f"Training RMSE : {train_rmse:.6f}")
print(f"Y target range: {Y.min():.3f} to {Y.max():.3f}")
print(f"Y pred range  : {Y_pred_train.min():.3f} to {Y_pred_train.max():.3f}")

# =============================================================
# OPTIONAL INDEPENDENT TEST DATASET EVALUATION
# =============================================================

test_mse = None
test_rmse = None
train_test_ratio = None

if USE_INDEPENDENT_TEST:

    with open(TEST_DATASET_FILE, "rb") as f_test:
        test_data = pickle.load(f_test)

    X_test = test_data["X"]
    Y_test = test_data["Y"]

    if X_test.shape[1] != X.shape[1]:
        raise ValueError(
            f"Input feature mismatch: training X has {X.shape[1]} features, "
            f"test X has {X_test.shape[1]} features."
        )

    if Y_test.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Output feature mismatch: training Y has {Y.shape[1]} features, "
            f"test Y has {Y_test.shape[1]} features."
        )

    if test_data["n_harmonics"] != n_harmonics:
        raise ValueError("Independent test dataset has different n_harmonics.")

    if "case_summaries" in test_data:
        df_cases = pd.DataFrame(test_data["case_summaries"])

        print("\nIndependent test case summaries:")
        print(df_cases.head(20))

    X_test_norm = (X_test - X_mean) / X_std
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        Y_pred_test_norm = model(X_test_tensor).numpy()

    Y_pred_test = Y_pred_test_norm * Y_std + Y_mean

    test_mse = np.mean((Y_pred_test - Y_test) ** 2)
    test_rmse = np.sqrt(test_mse)
    train_test_ratio = test_mse / train_mse

    print("\n" + "=" * 60)
    print("INDEPENDENT TEST GENERALIZATION CHECK")
    print("=" * 60)
    print(f"Training MSE          : {train_mse:.6e}")
    print(f"Independent Test MSE  : {test_mse:.6e}")
    print(f"Test / Train ratio    : {train_test_ratio:.2f}")
    print(f"Training RMSE         : {train_rmse:.6f}")
    print(f"Independent Test RMSE : {test_rmse:.6f}")

# =============================================================
# PREDICTED VS TARGET PLOTS
# =============================================================

plt.figure(figsize=(6, 5))
plt.scatter(Y.flatten(), Y_pred_train.flatten(), alpha=0.3, s=10)

min_train = min(Y.min(), Y_pred_train.min())
max_train = max(Y.max(), Y_pred_train.max())

plt.plot([min_train, max_train], [min_train, max_train], "r--")
plt.title("Training Dataset\nPredicted Final-Target Correction vs Target")
plt.xlabel("Target final correction coefficient")
plt.ylabel("Predicted final correction coefficient")
plt.grid(True)
plt.tight_layout()
plt.show()

if USE_INDEPENDENT_TEST:

    plt.figure(figsize=(6, 5))
    plt.scatter(Y_test.flatten(), Y_pred_test.flatten(), alpha=0.3, s=10)

    min_test = min(Y_test.min(), Y_pred_test.min())
    max_test = max(Y_test.max(), Y_pred_test.max())

    plt.plot([min_test, max_test], [min_test, max_test], "r--")
    plt.title("Independent Test Dataset\nPredicted vs Target")
    plt.xlabel("Target final correction coefficient")
    plt.ylabel("Predicted final correction coefficient")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =============================================================
# HARMONIC-BY-HARMONIC GENERALIZATION CHECK
# =============================================================

plt.figure(figsize=(14, 12))

rows = int(np.ceil(n_harmonics / 3))

for h in range(n_harmonics):

    plt.subplot(rows, 3, h + 1)

    idx_real = 2*h
    idx_imag = 2*h + 1

    plt.scatter(
        Y[:, idx_real],
        Y_pred_train[:, idx_real],
        alpha=0.4,
        label="Real",
        s=10
    )

    plt.scatter(
        Y[:, idx_imag],
        Y_pred_train[:, idx_imag],
        alpha=0.4,
        label="Imag",
        s=10
    )

    ymin = min(
        Y[:, idx_real].min(),
        Y_pred_train[:, idx_real].min(),
        Y[:, idx_imag].min(),
        Y_pred_train[:, idx_imag].min()
    )

    ymax = max(
        Y[:, idx_real].max(),
        Y_pred_train[:, idx_real].max(),
        Y[:, idx_imag].max(),
        Y_pred_train[:, idx_imag].max()
    )

    plt.plot([ymin, ymax], [ymin, ymax], "r--")
    plt.title(f"Harmonic {h + 1}")
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.grid(True)

    if h == 0:
        plt.legend()

plt.tight_layout()
plt.show()

# =============================================================
# NN FINAL-TARGET UPDATE FUNCTION
# =============================================================

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
# FFT-ILC ONE-STEP UPDATE FOR BASELINE DIAGNOSTIC ONLY
# =============================================================

def true_fft_ilc_correction(error_best, voltage_best, alpha):
    E_fft = np.fft.rfft(error_best)
    V_fft = np.fft.rfft(voltage_best)
    V_fft_new = V_fft.copy()

    for h in range(1, n_harmonics + 1):
        idx = np.argmin(np.abs(freqs - h * f))
        V_fft_new[idx] += alpha * E_fft[idx]

    voltage_true = np.fft.irfft(V_fft_new, n=len(t))
    voltage_true = np.clip(voltage_true, -V_CLIP, V_CLIP)

    return voltage_true - voltage_best


# =============================================================
# PHASE 4 — ONE-PREDICTION TEST
# =============================================================

print("\n" + "=" * 60)
print("PHASE 4A: ONE-PREDICTION FINAL-TARGET TEST")
print("=" * 60)

voltage_0 = V_amp * np.sin(2 * np.pi * f * t)
current_0 = run_qspice(voltage_0, runner, t)

error_0 = i_reference - current_0
rms_0 = rms_error(current_0)
thd_0 = calculate_thd(current_0, f, dt)

print(f"Initial RMS : {rms_0:.6f} A")
print(f"Initial THD : {thd_0:.3f}%")

# scale_feature is the same slot where alpha used to be in your X vector.
# In the new final-target dataset, this value came from the ILC alpha during data generation.
# We keep 0.4 as a reasonable representative value.
scale_feature_debug = 0.4

candidate_scales = [0.25, 0.50, 0.75, 1.00]

one_step_results = []

for scale in candidate_scales:

    voltage_candidate, correction_candidate, y_candidate = nn_final_target_voltage_update(
        error_signal=error_0,
        voltage_current=voltage_0,
        model=model,
        f=f,
        freqs=freqs,
        n_harmonics=n_harmonics,
        X_mean=X_mean,
        X_std=X_std,
        Y_mean=Y_mean,
        Y_std=Y_std,
        t=t,
        scale_feature=scale_feature_debug,
        V_amp=V_amp,
        I_target=I_target,
        LVAL=TEST_LVAL,
        ISAT=TEST_ISAT,
        LSAT=TEST_LSAT,
        apply_scale=scale,
        verbose=(scale == candidate_scales[0])
    )

    current_candidate = run_qspice(voltage_candidate, runner, t)
    error_candidate = i_reference - current_candidate
    rms_candidate = np.sqrt(np.mean(error_candidate**2))
    thd_candidate = calculate_thd(current_candidate, f, dt)

    one_step_results.append({
        "scale": scale,
        "voltage": voltage_candidate,
        "current": current_candidate,
        "correction": correction_candidate,
        "rms": rms_candidate,
        "thd": thd_candidate
    })

    print(
        f"Scale {scale:.2f} | "
        f"RMS {rms_candidate:.6f} A | "
        f"THD {thd_candidate:.3f}% | "
        f"improvement {rms_0 - rms_candidate:.6f} A"
    )

best_one_step = min(one_step_results, key=lambda d: d["rms"])

voltage_1 = best_one_step["voltage"]
current_1 = best_one_step["current"]
rms_1 = best_one_step["rms"]
thd_1 = best_one_step["thd"]
best_one_step_scale = best_one_step["scale"]

print("\n" + "=" * 60)
print("ONE-PREDICTION SUMMARY")
print("=" * 60)
print(f"Initial RMS       : {rms_0:.6f} A")
print(f"Best updated RMS  : {rms_1:.6f} A")
print(f"RMS improvement   : {rms_0 - rms_1:.6f} A")
print(f"Initial THD       : {thd_0:.3f}%")
print(f"Best updated THD  : {thd_1:.3f}%")
print(f"Best scale        : {best_one_step_scale:.2f}")

plt.figure(figsize=(10, 4))
plt.plot(t, current_0, label="Initial current")
plt.plot(t, current_1, label=f"After NN final-target update, scale={best_one_step_scale}")
plt.plot(t, i_reference, "--", label="Reference")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("One-Prediction Final-Target NN Test")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================
# DIAGNOSTIC: NN FINAL-TARGET CORRECTION VS FFT ONE-STEP CORRECTION
# =============================================================

print("\n" + "=" * 60)
print("FINAL-TARGET NN VS FFT ONE-STEP DIAGNOSTIC")
print("=" * 60)

true_fft_corr = true_fft_ilc_correction(
    error_0,
    voltage_0,
    alpha=0.4
)

nn_final_corr = best_one_step["correction"]

cos_fft_vs_nn = safe_cosine(true_fft_corr, nn_final_corr)

print(f"FFT one-step correction RMS : {np.sqrt(np.mean(true_fft_corr**2)):.6f} V")
print(f"NN final-target corr RMS    : {np.sqrt(np.mean(nn_final_corr**2)):.6f} V")
print(f"Cosine similarity           : {cos_fft_vs_nn:.4f}")

print(
    "\nNote: cosine does NOT need to be close to 1 anymore. "
    "The NN is no longer trying to copy one FFT step."
)

plt.figure(figsize=(10, 4))
plt.plot(t, true_fft_corr, label="FFT one-step correction")
plt.plot(t, nn_final_corr, label="NN predicted final-target correction", alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Voltage correction (V)")
plt.title("FFT One-Step vs NN Final-Target Correction")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================
# PHASE 4B — CLOSED-LOOP NN VALIDATION
# =============================================================

print("\n" + "=" * 60)
print("PHASE 4B: CLOSED-LOOP FINAL-TARGET NN VALIDATION")
print("=" * 60)

voltage_nn = V_amp * np.sin(2 * np.pi * f * t)
voltage_best_nn = voltage_nn.copy()
error_best_nn = np.zeros(len(t))

rms_best_nn = np.inf
thd_best_nn = np.inf

consecutive_fails_nn = 0
k_nn = 0

nn_scale = NN_SCALE_INIT

nn_error_history = []
nn_voltage_history = []
nn_current_history = []
nn_rms_history = []
nn_thd_history = []
nn_scale_history = []
nn_correction_rms_history = []
nn_correction_peak_history = []
nn_clipped_fraction_history = []

while True:

    k_nn += 1

    if k_nn > max_iterations:
        print(f"\nNN stopped: max iterations reached ({max_iterations}).")
        break

    print(f"\nNN Iteration {k_nn}")

    current_nn = run_qspice(voltage_nn, runner, t)
    error_nn = i_reference - current_nn
    rms_nn = np.sqrt(np.mean(error_nn**2))
    thd_nn = calculate_thd(current_nn, f, dt)

    nn_error_history.append(error_nn.copy())
    nn_voltage_history.append(voltage_nn.copy())
    nn_current_history.append(current_nn.copy())
    nn_rms_history.append(rms_nn)
    nn_thd_history.append(thd_nn)
    nn_scale_history.append(nn_scale)

    if k_nn == 1:
        error_features_phase4 = extract_harmonic_features(
            error_nn,
            f,
            freqs,
            n_harmonics
        )

        voltage_features_phase4 = extract_harmonic_features(
            voltage_nn,
            f,
            freqs,
            n_harmonics
        )

        print("\nPHASE 4 DISTRIBUTION CHECK")
        print(f"  Error feature range   : {error_features_phase4.min():.3f} to {error_features_phase4.max():.3f}")
        print(f"  Voltage feature range : {voltage_features_phase4.min():.3f} to {voltage_features_phase4.max():.3f}")
        print(f"  Training X range      : {X.min():.3f} to {X.max():.3f}")
        print(f"  Initial RMS           : {rms_nn:.6f} A")
        print(f"  Initial THD           : {thd_nn:.3f}%")

    print(
        f"  RMS {rms_nn:.6f} A | "
        f"THD {thd_nn:.3f}% | "
        f"scale {nn_scale:.3f} | "
        f"Vpk {np.max(np.abs(voltage_nn)):.3f} V"
    )

    if rms_nn <= comparison_target:

        rms_best_nn = rms_nn
        thd_best_nn = thd_nn
        voltage_best_nn = voltage_nn.copy()
        error_best_nn = error_nn.copy()

        print(
            f"\nNN target reached: "
            f"{rms_nn:.6f} A <= {comparison_target:.6f} A"
        )
        break

    if rms_nn < rms_best_nn - 1e-6:

        rms_best_nn = rms_nn
        thd_best_nn = thd_nn
        voltage_best_nn = voltage_nn.copy()
        error_best_nn = error_nn.copy()
        consecutive_fails_nn = 0

        old_scale = nn_scale
        nn_scale = min(NN_SCALE_MAX, nn_scale * NN_SCALE_ACCEPT_FACTOR)

        print(
            f"  Accepted | best RMS {rms_best_nn:.6f} | "
            f"scale {old_scale:.3f} -> {nn_scale:.3f}"
        )

    else:

        consecutive_fails_nn += 1

        old_scale = nn_scale
        nn_scale = max(NN_SCALE_MIN, nn_scale * NN_SCALE_FAIL_FACTOR)

        print(
            f"  Rejected | best RMS {rms_best_nn:.6f} | "
            f"scale {old_scale:.3f} -> {nn_scale:.3f} | "
            f"fails {consecutive_fails_nn}/{max_consecutive_fails}"
        )

        voltage_nn = voltage_best_nn.copy()
        error_nn = error_best_nn.copy()

        if consecutive_fails_nn >= max_consecutive_fails:
            print(f"\nNN stopping: {max_consecutive_fails} consecutive fails.")
            break

    # Predict final-target correction from best accepted state
    voltage_candidate, correction_candidate, y_candidate = nn_final_target_voltage_update(
        error_signal=error_best_nn,
        voltage_current=voltage_best_nn,
        model=model,
        f=f,
        freqs=freqs,
        n_harmonics=n_harmonics,
        X_mean=X_mean,
        X_std=X_std,
        Y_mean=Y_mean,
        Y_std=Y_std,
        t=t,
        scale_feature=scale_feature_debug,
        V_amp=V_amp,
        I_target=I_target,
        LVAL=TEST_LVAL,
        ISAT=TEST_ISAT,
        LSAT=TEST_LSAT,
        apply_scale=nn_scale,
        verbose=False
    )

    unclipped_voltage = voltage_best_nn + nn_scale * correction_candidate
    clipped_fraction = np.mean(np.abs(unclipped_voltage) > V_CLIP)

    nn_correction_rms_history.append(
        np.sqrt(np.mean(correction_candidate**2))
    )

    nn_correction_peak_history.append(
        np.max(np.abs(correction_candidate))
    )

    nn_clipped_fraction_history.append(
        clipped_fraction
    )

    print(
        f"  Predicted final-target correction RMS: "
        f"{nn_correction_rms_history[-1]:.6f} V"
    )

    print(
        f"  Predicted final-target correction peak: "
        f"{nn_correction_peak_history[-1]:.6f} V"
    )

    print(
        f"  Clipped fraction if applied: "
        f"{100 * clipped_fraction:.2f}%"
    )

    voltage_nn = voltage_candidate

# =============================================================
# CLOSED-LOOP SUMMARY
# =============================================================

best_idx = int(np.argmin(nn_rms_history))

print("\n" + "=" * 60)
print("FULL FINAL-TARGET NN DIAGNOSTIC FINISHED")
print("=" * 60)

print(f"Iterations run        : {k_nn}")
print(f"Best RMS              : {nn_rms_history[best_idx]:.6f} A")
print(f"Best THD              : {nn_thd_history[best_idx]:.4f}%")
print(f"Best iteration        : {best_idx + 1}")
print(f"Final RMS             : {nn_rms_history[-1]:.6f} A")
print(f"Final THD             : {nn_thd_history[-1]:.4f}%")
print(f"Final scale           : {nn_scale:.4f}")

for i, rms in enumerate(nn_rms_history, start=1):
    if rms <= comparison_target:
        print(
            f"NN reached RMS <= {comparison_target:.3f} "
            f"at iteration {i}"
        )
        break
else:
    print(
        f"NN never reached RMS <= {comparison_target:.3f}"
    )

# =============================================================
# PLOTS: CLOSED LOOP
# =============================================================

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(nn_rms_history) + 1), nn_rms_history, label="NN RMS")
plt.axhline(comparison_target, linestyle="--", label=f"Target RMS = {comparison_target}")
plt.xlabel("Iteration")
plt.ylabel("RMS Error (A)")
plt.title("Closed-Loop Final-Target NN RMS Convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(nn_thd_history) + 1), nn_thd_history, label="NN THD")
plt.xlabel("Iteration")
plt.ylabel("THD (%)")
plt.title("Closed-Loop Final-Target NN THD")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(nn_scale_history) + 1), nn_scale_history, label="NN scale")
plt.xlabel("Iteration")
plt.ylabel("Scale")
plt.title("NN Final-Target Scale History")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, nn_current_history[0], label="Initial current")
plt.plot(t, nn_current_history[best_idx], label=f"Best NN current, iter {best_idx + 1}")
plt.plot(t, i_reference, "--", label="Reference")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("Best Closed-Loop NN Current Waveform")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, nn_voltage_history[0], label="Initial voltage")
plt.plot(t, nn_voltage_history[best_idx], label=f"Best NN voltage, iter {best_idx + 1}")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Initial vs Best NN Voltage Waveform")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================
# MODEL ACCEPTANCE CHECK
# =============================================================

if USE_INDEPENDENT_TEST:
    generalization_ok = train_test_ratio < 3.0
else:
    generalization_ok = True

one_step_ok = rms_1 < rms_0
closed_loop_ok = nn_rms_history[best_idx] <= comparison_target

# For final-target learning, cosine with FFT one-step is no longer required.
# It can even be low, because the NN is not supposed to copy one FFT step.
good_model = (
    generalization_ok and
    one_step_ok and
    closed_loop_ok
)

print("\n" + "=" * 60)
print("MODEL ACCEPTANCE CHECK")
print("=" * 60)

print(f"Generalization OK : {generalization_ok}")
if USE_INDEPENDENT_TEST:
    print(f"Test/train ratio  : {train_test_ratio:.2f}")
else:
    print("Test/train ratio  : skipped, no independent final-target test dataset")

print(f"One-step RMS      : {rms_0:.6f} -> {rms_1:.6f}")
print(f"Best closed RMS   : {nn_rms_history[best_idx]:.6f}")
print(f"FFT cosine diag   : {cos_fft_vs_nn:.4f}  <-- diagnostic only now")

if good_model:
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"\nMODEL ACCEPTED AND SAVED TO {MODEL_FILE}")
else:
    print("\nMODEL REJECTED")