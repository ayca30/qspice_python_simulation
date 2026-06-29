from QSPICEBatchRunner import QSPICEBatchRunner
import numpy as np
import os
import pickle

# =============================================================
# SETUP
# =============================================================

runner = QSPICEBatchRunner(
    basefile="Controller_Test_Diverse_Training",
    workdir="ILC_FINAL_TARGET_DATASET"
)

runner.qsch_to_cir(
    r"C:\Users\aycah\Documents\RISE\qspice_and_training\NEURAL_NETWORK_TRAINING\from VM\NN_2.0\Controller_Test_Diverse_Training.qsch"
)

# =============================================================
# SWEEP SETTINGS
# =============================================================

frequencies = [50, 100, 300, 700, 1000]
V_amps = [0.5, 1.0, 2.0]
alpha_inits = [0.1, 0.4]

L_values = [0.5e-3, 2e-3, 10e-3]
ISAT_values = [1, 5, 10]
LSAT_values = [5e-6, 20e-6]

I_targets = [1.0, 2.0, 2.5]

# =============================================================
# FIXED SETTINGS
# =============================================================

rms_target = 0.05
RMS_DATASET_LIMIT = 0.05

max_iterations = 500
max_consecutive_fails = 40
rms_improvement_tol = 1e-5

dt = 1e-6
n_harmonics = 12

alpha_min = 0.005
alpha_max = 0.9
alpha_fail_factor = 0.9
alpha_recovery_factor = 1.05

V_CLIP = 10
RVAL = 1.0

CLEANUP_REQUIRED = 10

OUTPUT_FILE = "multi_ilc_dataset_final_target_12h.pkl"
PARTIAL_FILE = "multi_ilc_dataset_final_target_12h_partial.pkl"

# =============================================================
# HELPERS
# =============================================================

def extract_harmonic_features(signal, f, freqs, n_harmonics):
    spectrum = np.fft.rfft(signal)
    features = []

    for h in range(1, n_harmonics + 1):
        idx = np.argmin(np.abs(freqs - h * f))
        features.append(spectrum[idx].real)
        features.append(spectrum[idx].imag)

    return np.array(features, dtype=np.float32)


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


def run_qspice(voltage, runner, t, LVAL, ISAT, LSAT, RVAL):
    from scipy.interpolate import interp1d

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
# SINGLE CASE GENERATION
# =============================================================

def run_ilc_case(
    case_id,
    f,
    V_amp,
    I_target,
    alpha_init,
    LVAL,
    ISAT,
    LSAT,
    runner
):

    t = np.arange(0, 1 / f, dt)
    freqs = np.fft.rfftfreq(len(t), d=dt)

    i_reference = I_target * np.sin(2 * np.pi * f * t)
    voltage = V_amp * np.sin(2 * np.pi * f * t)

    alpha = alpha_init

    rms_history = []
    thd_history = []
    alpha_history = []

    accepted_snapshots = []

    rms_best = np.inf
    voltage_best = voltage.copy()
    error_best = np.zeros(len(t))

    consecutive_fails = 0
    cleanup_counter = 0

    converged = False

    print("\n" + "=" * 70)
    print(f"CASE {case_id}")
    print(
        f"f={f} Hz | V_amp={V_amp} | I_target={I_target} | "
        f"alpha_init={alpha_init} | L={LVAL} H | "
        f"ISAT={ISAT} | LSAT={LSAT} H"
    )
    print("=" * 70)

    for k in range(1, max_iterations + 1):

        current = run_qspice(
            voltage,
            runner,
            t,
            LVAL,
            ISAT,
            LSAT,
            RVAL
        )

        error = i_reference - current
        rms = np.sqrt(np.mean(error**2))
        thd = calculate_thd(current, f, dt)

        rms_history.append(rms)
        thd_history.append(thd)
        alpha_history.append(alpha)

        print(
            f"  Iter {k:04d} | "
            f"RMS {rms:.6f} A | "
            f"THD {thd:.3f}% | "
            f"alpha {alpha:.5f}"
        )

        if rms <= rms_target:
            cleanup_counter += 1
            print(f"    Cleanup {cleanup_counter}/{CLEANUP_REQUIRED}")

            if cleanup_counter >= CLEANUP_REQUIRED:
                converged = True
                print(f"  CONVERGED: best RMS {rms_best:.6f} A")
                break
        else:
            cleanup_counter = 0

        if rms < rms_best - rms_improvement_tol:

            rms_best = rms
            voltage_best = voltage.copy()
            error_best = error.copy()
            consecutive_fails = 0

            # Store accepted state temporarily.
            # We do NOT create Y yet because we do not know voltage_final yet.
            accepted_snapshots.append({
                "error": error_best.copy(),
                "voltage": voltage_best.copy(),
                "alpha": alpha,
                "rms": rms_best,
                "iteration": k
            })

            old_alpha = alpha
            alpha = min(alpha_max, alpha * alpha_recovery_factor)

            print(
                f"    Accepted | best RMS {rms_best:.6f} | "
                f"alpha {old_alpha:.5f} -> {alpha:.5f}"
            )

        else:
            consecutive_fails += 1

            old_alpha = alpha
            alpha = max(alpha_min, alpha * alpha_fail_factor)

            print(
                f"    Rejected | best RMS {rms_best:.6f} | "
                f"alpha {old_alpha:.5f} -> {alpha:.5f} | "
                f"fails {consecutive_fails}/{max_consecutive_fails}"
            )

            voltage = voltage_best.copy()
            error = error_best.copy()

            if consecutive_fails >= max_consecutive_fails:
                print(f"  Stopping: {max_consecutive_fails} consecutive fails.")
                break

        # FFT-ILC update
        E_fft = np.fft.rfft(error_best)
        V_fft = np.fft.rfft(voltage_best)
        V_fft_new = V_fft.copy()

        for h in range(1, n_harmonics + 1):
            idx = np.argmin(np.abs(freqs - h * f))
            V_fft_new[idx] += alpha * E_fft[idx]

        voltage = np.fft.irfft(V_fft_new, n=len(t))
        voltage = np.clip(voltage, -V_CLIP, V_CLIP)

    # =============================================================
    # ACCEPT / REJECT CASE
    # =============================================================

    best_rms = min(rms_history)
    best_idx = int(np.argmin(rms_history))

    accepted_for_dataset = (
        converged and
        best_rms <= RMS_DATASET_LIMIT and
        len(accepted_snapshots) > 0
    )

    X_case = []
    Y_case = []

    if accepted_for_dataset:

        voltage_final = voltage_best.copy()

        print(
            f"  CASE ACCEPTED FOR DATASET | "
            f"best RMS = {best_rms:.6f} A | "
            f"snapshots = {len(accepted_snapshots)}"
        )

        for snap in accepted_snapshots:

            error_features = extract_harmonic_features(
                snap["error"],
                f,
                freqs,
                n_harmonics
            )

            voltage_features = extract_harmonic_features(
                snap["voltage"],
                f,
                freqs,
                n_harmonics
            )

            alpha_feature = np.array(
                [snap["alpha"]],
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
                alpha_feature,
                plant_features
            ])

            # =====================================================
            # IMPORTANT NEW TARGET
            # =====================================================
            # Instead of one-step correction:
            #     V_next - V_current
            #
            # We train toward the final converged voltage:
            #     V_final - V_current
            # =====================================================

            final_voltage_correction = voltage_final - snap["voltage"]

            y = extract_harmonic_features(
                final_voltage_correction,
                f,
                freqs,
                n_harmonics
            )

            X_case.append(x)
            Y_case.append(y)

    else:
        print(
            f"  CASE REJECTED | "
            f"converged={converged} | "
            f"best RMS={best_rms:.6f} A | "
            f"snapshots={len(accepted_snapshots)}"
        )

    return {
        "case_id": case_id,

        "accepted_for_dataset": accepted_for_dataset,

        "X": X_case,
        "Y": Y_case,

        "rms_history": rms_history,
        "thd_history": thd_history,
        "alpha_history": alpha_history,

        "dt": dt,
        "n_harmonics": n_harmonics,

        "f": f,
        "V_amp": V_amp,
        "I_target": I_target,
        "rms_target": rms_target,

        "LVAL": LVAL,
        "ISAT": ISAT,
        "LSAT": LSAT,
        "RVAL": RVAL,
        "alpha_init": alpha_init,

        "iterations": len(rms_history),
        "final_rms": rms_history[-1],
        "final_thd": thd_history[-1],
        "best_rms": best_rms,
        "best_iteration": best_idx + 1,
        "num_training_samples": len(X_case),
        "converged": converged,
    }


# =============================================================
# RUN FULL SWEEP
# =============================================================

all_X = []
all_Y = []
case_summaries = []

case_id = 0
accepted_cases = 0
rejected_cases = 0

total_cases = (
    len(frequencies)
    * len(V_amps)
    * len(I_targets)
    * len(alpha_inits)
    * len(L_values)
    * len(ISAT_values)
    * len(LSAT_values)
)

print("\n" + "=" * 70)
print("STARTING FINAL-TARGET ILC DATASET GENERATION")
print(f"Total cases: {total_cases}")
print(f"n_harmonics: {n_harmonics}")
print(f"Dataset RMS limit: {RMS_DATASET_LIMIT}")
print("=" * 70)

for f_case in frequencies:
    for V_amp_case in V_amps:
        for I_target_case in I_targets:
            for alpha_case in alpha_inits:
                for L_case in L_values:
                    for ISAT_case in ISAT_values:
                        for LSAT_case in LSAT_values:

                            case_id += 1

                            try:
                                case_data = run_ilc_case(
                                    case_id=case_id,
                                    f=f_case,
                                    V_amp=V_amp_case,
                                    I_target=I_target_case,
                                    alpha_init=alpha_case,
                                    LVAL=L_case,
                                    ISAT=ISAT_case,
                                    LSAT=LSAT_case,
                                    runner=runner
                                )

                                if case_data["accepted_for_dataset"]:

                                    accepted_cases += 1

                                    all_X.extend(case_data["X"])
                                    all_Y.extend(case_data["Y"])

                                else:
                                    rejected_cases += 1

                                case_summaries.append({
                                    "case_id": case_data["case_id"],
                                    "accepted_for_dataset": case_data["accepted_for_dataset"],
                                    "converged": case_data["converged"],

                                    "f": case_data["f"],
                                    "V_amp": case_data["V_amp"],
                                    "I_target": case_data["I_target"],
                                    "alpha_init": case_data["alpha_init"],

                                    "LVAL": case_data["LVAL"],
                                    "ISAT": case_data["ISAT"],
                                    "LSAT": case_data["LSAT"],
                                    "RVAL": case_data["RVAL"],

                                    "iterations": case_data["iterations"],
                                    "best_iteration": case_data["best_iteration"],
                                    "final_rms": case_data["final_rms"],
                                    "final_thd": case_data["final_thd"],
                                    "best_rms": case_data["best_rms"],

                                    "num_training_samples": case_data["num_training_samples"],
                                })

                            except Exception as e:
                                rejected_cases += 1

                                print("\nERROR IN CASE")
                                print(f"case_id = {case_id}")
                                print(f"f = {f_case}")
                                print(f"V_amp = {V_amp_case}")
                                print(f"I_target = {I_target_case}")
                                print(f"alpha_init = {alpha_case}")
                                print(f"LVAL = {L_case}")
                                print(f"ISAT = {ISAT_case}")
                                print(f"LSAT = {LSAT_case}")
                                print(f"Error: {e}")

                            # Partial save every 10 cases
                            if case_id % 10 == 0:

                                partial_data = {
                                    "X": np.array(all_X, dtype=np.float32),
                                    "Y": np.array(all_Y, dtype=np.float32),

                                    "case_summaries": case_summaries,

                                    "completed_cases": case_id,
                                    "total_cases": total_cases,

                                    "description": (
                                        "Partial final-target ILC dataset. "
                                        "Y = harmonic features of V_final - V_current. "
                                        "Only cases with best_rms <= 0.05 are included."
                                    ),

                                    "dt": dt,
                                    "n_harmonics": n_harmonics,
                                    "rms_target": rms_target,
                                    "rms_dataset_limit": RMS_DATASET_LIMIT,

                                    "sweep_settings": {
                                        "frequencies": frequencies,
                                        "V_amps": V_amps,
                                        "I_targets": I_targets,
                                        "alpha_inits": alpha_inits,
                                        "L_values": L_values,
                                        "ISAT_values": ISAT_values,
                                        "LSAT_values": LSAT_values,
                                    }
                                }

                                with open(PARTIAL_FILE, "wb") as f_out:
                                    pickle.dump(partial_data, f_out)

                                print(
                                    f"\nPartial save complete: "
                                    f"{case_id}/{total_cases} cases attempted | "
                                    f"accepted={accepted_cases} | "
                                    f"rejected={rejected_cases} | "
                                    f"samples={len(all_X)}"
                                )

# =============================================================
# SAVE FINAL DATASET
# =============================================================

multi_ilc_data = {
    "X": np.array(all_X, dtype=np.float32),
    "Y": np.array(all_Y, dtype=np.float32),

    "case_summaries": case_summaries,

    "completed_cases": case_id,
    "total_cases": total_cases,

    "description": (
        "Final-target multi-parameter ILC dataset for NN training. "
        "Inputs contain error harmonics, voltage harmonics, alpha, and plant features. "
        "Targets are harmonic features of V_final - V_current, where V_final is the "
        "best converged voltage from accepted FFT-ILC cases only."
    ),

    "dt": dt,
    "n_harmonics": n_harmonics,
    "rms_target": rms_target,
    "rms_dataset_limit": RMS_DATASET_LIMIT,

    "accepted_cases": accepted_cases,
    "rejected_cases": rejected_cases,

    "feature_description": {
        "X": [
            "error harmonic features: Re/Im for harmonics 1..n",
            "voltage harmonic features: Re/Im for harmonics 1..n",
            "alpha",
            "plant features: f/1000, V_amp/4, I_target/2.5, LVAL/0.01, ISAT/10, LSAT/50e-6"
        ],
        "Y": [
            "final voltage correction harmonic features",
            "Y = harmonic_features(V_final - V_current)"
        ]
    },

    "sweep_settings": {
        "frequencies": frequencies,
        "V_amps": V_amps,
        "I_targets": I_targets,
        "alpha_inits": alpha_inits,
        "L_values": L_values,
        "ISAT_values": ISAT_values,
        "LSAT_values": LSAT_values,
    }
}

with open(OUTPUT_FILE, "wb") as f_out:
    pickle.dump(multi_ilc_data, f_out)

print("\n" + "=" * 70)
print("FINAL DATASET COMPLETE")
print("=" * 70)
print(f"Cases attempted       : {case_id}")
print(f"Accepted cases        : {accepted_cases}")
print(f"Rejected cases        : {rejected_cases}")
print(f"Training samples saved: {len(all_X)}")
print(f"X shape               : {multi_ilc_data['X'].shape}")
print(f"Y shape               : {multi_ilc_data['Y'].shape}")
print(f"Saved to              : {OUTPUT_FILE}")