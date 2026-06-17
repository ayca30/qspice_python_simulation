from QSPICEBatchRunner import QSPICEBatchRunner
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil



# -----------------------------
# SETUP
# -----------------------------

runner = QSPICEBatchRunner(
    basefile="Controller_Test_PWL_ILC",
    workdir="ILC_test_2"
)

runner.qsch_to_cir(
    r"C:\Users\aycah\Documents\RISE\qspice_and_training\Controller_Test_PWL_ILC.qsch"
)

# -----------------------------------
# SETTINGS
# -----------------------------------

f = 100
rms_target = 0.02          # 2% RMS error target
max_consecutive_fails = 15
rms_improvement_tol = 1e-4

dt = 1e-5
t = np.arange(0, 1/f, dt)


# desired current

I_target = 1

i_reference = (
    I_target *
    np.sin(2*np.pi*f*t)
)


# initial voltage guess

V_amp = 3

voltage = (
    V_amp *
    np.sin(2*np.pi*f*t)
)


# -----------------------------------
# ADAPTIVE LEARNING RATE SETTINGS
# -----------------------------------

alpha = 0.3               # starting learning rate
alpha_min = 0.005         # floor — never goes below this
alpha_max = 0.5           # ceiling — never goes above this
alpha_fail_factor = 0.7   # multiply alpha by this on failure
alpha_recovery_factor = 1.01  # multiply alpha by this on success


# -----------------------------------
# THD FUNCTION
# -----------------------------------

def calculate_thd(signal, f, dt):

    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=dt)
    spectrum = np.abs(np.fft.rfft(signal)) / N

    f1_idx = np.argmin(np.abs(freqs - f))
    V1 = spectrum[f1_idx]

    harmonics = []
    for h in range(2, 10):
        fh_idx = np.argmin(np.abs(freqs - h * f))
        harmonics.append(spectrum[fh_idx])

    thd = np.sqrt(sum(v**2 for v in harmonics)) / V1
    return thd * 100  # percentage

# -----------------------------------
# FFT SPECTRUM PLOT
# -----------------------------------

def plot_fft(signal, dt, title):

    N = len(signal)

    freqs = np.fft.rfftfreq(
        N,
        d=dt
    )

    spectrum = (
        np.abs(np.fft.rfft(signal))
        / N
    )


    plt.figure(figsize=(10,5))

    plt.stem(
        freqs,
        spectrum
    )

    plt.xlim(
        0,
        1000
    )

    plt.title(title)

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Magnitude"
    )

    plt.grid()



# store history

error_history = []
voltage_history = []
current_history = []
thd_history = []
rms_history = []
alpha_history = []


# -----------------------------------
# ALGORITHM STATE
# -----------------------------------

rms_best = np.inf
voltage_best = voltage.copy()
error_best = np.zeros(len(t))
consecutive_fails = 0
max_consecutive_fails = 15



# -----------------------------------
# ILC LOOP
# -----------------------------------
k = 0

while True:

    k += 1

    print(f"\nIteration {k}")

    # -------------------------------
    # WRITE VOLTAGE FILE FOR QSPICE
    # -------------------------------

    voltage_file = os.path.join(
        runner.workdir,
        "voltage.txt"
    )

    np.savetxt(
        voltage_file,
        np.column_stack((t, voltage))
    )

    print("Voltage file created")


    # -------------------------------
    # RUN QSPICE
    # -------------------------------

    cir_files = runner.generate_param_cir_files(
        [
            {}
        ]
    )

    result = runner.run_batch(
        cir_files,
        signals=["I(L1)"],
        max_workers=1
    )[0]

    df = result["data"]

    time_sim = df.iloc[:,0].to_numpy()

    current = (
        df["I(L1)"]
        .to_numpy()
    )

    # match lengths
    current = current[:len(t)]


    # -------------------------------
    # ERROR
    # -------------------------------

    error = (
        i_reference -
        current
    )

    rms = np.sqrt(np.mean(error**2))
    thd = calculate_thd(current, f, dt)

    print(f"RMS error: {rms:.4f} A")
    print(f"THD:       {thd:.2f}%")
    print(f"Alpha:     {alpha:.4f}")

        # -------------------------------
    # RMS TARGET CHECK
    # -------------------------------

    if rms <= rms_target:

        print(
            f"\nStopping early — RMS target reached "
            f"({rms:.4f} A <= {rms_target:.4f} A)"
        )

        voltage_history.append(voltage.copy())
        current_history.append(current.copy())
        error_history.append(error.copy())
        thd_history.append(thd)
        rms_history.append(rms)
        alpha_history.append(alpha)

        break

    # -------------------------------
    # ACCEPT / REJECT
    # -------------------------------

    if rms < rms_best - rms_improvement_tol:

        # improvement — accept
        rms_best = rms
        voltage_best = voltage.copy()
        error_best = error.copy()
        consecutive_fails = 0

        # alpha recovers slightly after success
        old_alpha = alpha
        alpha = min(alpha_max, alpha * alpha_recovery_factor)

        print(
            f"Accepted — new best RMS: {rms_best:.4f} A | "
            f"alpha {old_alpha:.4f} -> {alpha:.4f}"
        )

    else:

        # no improvement — reject, reduce alpha
        consecutive_fails += 1
        old_alpha = alpha
        alpha = max(alpha_min, alpha * alpha_fail_factor)

        print(
            f"Rejected — best RMS still: {rms_best:.4f} A | "
            f"alpha {old_alpha:.4f} -> {alpha:.4f} | "
            f"fails: {consecutive_fails}/{max_consecutive_fails}"
        )

        # roll back to best voltage and use best error for next update
        voltage = voltage_best.copy()
        error = error_best.copy()

        if consecutive_fails >= max_consecutive_fails:
            print(f"\nStopping early — {max_consecutive_fails} consecutive failed updates.")
            voltage_history.append(voltage.copy())
            current_history.append(current.copy())
            error_history.append(error.copy())
            thd_history.append(thd)
            rms_history.append(rms)
            alpha_history.append(alpha)
            break


    # -------------------------------
    # UPDATE VOLTAGE
    # -------------------------------

    voltage = (
        voltage_best +
        alpha * error_best
    )


    # save data

    voltage_history.append(voltage.copy())
    current_history.append(current.copy())
    error_history.append(error.copy())
    thd_history.append(thd)
    rms_history.append(rms)
    alpha_history.append(alpha)



# -----------------------------------
# PLOTS
# -----------------------------------

actual_iterations = len(rms_history)

# which iterations to plot
indices = list(range(0, actual_iterations, 3))
if (actual_iterations - 1) not in indices:
    indices.append(actual_iterations - 1)


# CURRENT

plt.figure(figsize=(12,5))

for k in indices:

    plt.plot(
        t,
        current_history[k],
        label=f"Iteration {k+1}"
    )

plt.plot(
    t,
    i_reference,
    "k--",
    label="Target"
)

plt.title("Current convergence during ILC")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.grid()
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=7,
    ncol=2
)
plt.tight_layout()


# ERROR

plt.figure(figsize=(12,5))

for k in indices:

    plt.plot(
        t,
        error_history[k],
        label=f"Iteration {k+1}"
    )

plt.title("Current waveform error")
plt.xlabel("Time (s)")
plt.ylabel("Error (A)")
plt.grid()
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=7,
    ncol=2
)
plt.tight_layout()


# VOLTAGE

plt.figure(figsize=(12,5))

for k in indices:

    plt.plot(
        t,
        voltage_history[k],
        label=f"Iteration {k+1}"
    )

plt.title("Voltage waveform updates")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid()
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=7,
    ncol=2
)
plt.tight_layout()


# RMS + THD + ALPHA VS ITERATION

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

ax1.plot(
    range(1, actual_iterations + 1),
    rms_history,
    marker='o'
)
ax1.set_title("RMS Error vs Iteration")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("RMS Error (A)")
ax1.grid()

ax2.plot(
    range(1, actual_iterations + 1),
    thd_history,
    marker='s',
    color='orange'
)
ax2.set_title("THD vs Iteration")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("THD (%)")
ax2.grid()

ax3.plot(
    range(1, actual_iterations + 1),
    alpha_history,
    marker='^',
    color='green'
)
ax3.set_title("Learning Rate (Alpha) vs Iteration")
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Alpha")
ax3.grid()

plt.tight_layout()

# -----------------------------------
# FFT HARMONIC ANALYSIS
# -----------------------------------

plot_fft(
    current_history[-1],
    dt,
    "Current FFT Spectrum - Final Iteration"
)


plot_fft(
    error_history[-1],
    dt,
    "Error FFT Spectrum - Final Iteration"
)

plt.show()