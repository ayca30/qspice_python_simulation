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
    workdir="ILC_test"
)

runner.qsch_to_cir(
    r"C:\Users\aycah\Documents\RISE\qspice_and_training\Controller_Test_PWL_ILC.qsch"
)

# -----------------------------------
# SETTINGS
# -----------------------------------

f = 100
iterations = 75

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


learning_rate = 0.5


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


# store history

error_history = []
voltage_history = []
current_history = []
thd_history = []



# -----------------------------------
# ILC LOOP
# -----------------------------------

for k in range(iterations):

    print(f"\nIteration {k+1}")


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
    print(f"THD: {thd:.2f}%")


    # -------------------------------
    # UPDATE VOLTAGE
    # -------------------------------


    voltage = (
        voltage +
        learning_rate*error
    )


    # save data

    voltage_history.append(voltage.copy())
    current_history.append(current.copy())
    error_history.append(error.copy())
    thd_history.append(thd)



# -----------------------------------
# PLOTS
# -----------------------------------

# which iterations to plot
indices = list(range(0, iterations, 3))
if (iterations - 1) not in indices:
    indices.append(iterations - 1)


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

plt.title(
    "Current convergence during ILC"
)

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

plt.title(
    "Current waveform error"
)

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

plt.title(
    "Voltage waveform updates"
)

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


# RMS + THD VS ITERATION

rms_per_iteration = [
    np.sqrt(np.mean(e**2))
    for e in error_history
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(
    range(1, iterations + 1),
    rms_per_iteration,
    marker='o'
)
ax1.set_title("RMS Error vs Iteration")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("RMS Error (A)")
ax1.grid()

ax2.plot(
    range(1, iterations + 1),
    thd_history,
    marker='s',
    color='orange'
)
ax2.set_title("THD vs Iteration")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("THD (%)")
ax2.grid()

plt.tight_layout()


plt.show()