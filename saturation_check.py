from QSPICEBatchRunner import QSPICEBatchRunner
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# SETUP
# -----------------------------

runner = QSPICEBatchRunner(
    basefile="Controller_Test",
    workdir="control_batch_6"
)

runner.qsch_to_cir(
    r"C:\Users\aycah\Documents\RISE\qspice_and_training\Controller_Test.qsch"
)


# -----------------------------
# VOLTAGE SWEEP
# -----------------------------

vamps = np.linspace(0.5, 15, 20)

param_list = [
    {"VAMP": v}
    for v in vamps
]


cir_files = runner.generate_param_cir_files(param_list)


results = runner.run_batch(
    cir_files,
    signals=["I(L1)"],
    max_workers=4
)


# -----------------------------
# SETTINGS
# -----------------------------

f = 100             # Hz
I_target_peak = 1   # desired current amplitude (A)

peak_currents = []


# store one example waveform for error calculation
best_t = None
best_i = None



# -----------------------------
# CURRENT WAVEFORM PLOT
# -----------------------------

plt.figure(figsize=(10,6))


for idx, result in enumerate(results):

    if result["status"] != "OK":
        continue

    df = result["data"]

    if df is None:
        continue


    t = df.iloc[:,0].to_numpy()
    i = df["I(L1)"].to_numpy()


    plt.plot(
        t,
        i,
        label=f"VAMP={vamps[idx]:.1f} V"
    )


    # save highest voltage waveform for error analysis
    if idx == len(results)-1:
        best_t = t
        best_i = i


    # last cycle peak
    T = 1/f

    mask = t >= (t[-1]-T)


    peak_currents.append(
        np.max(np.abs(i[mask]))
    )



plt.title(
    "Current waveform vs input amplitude\n"
    "Nonlinear inductor: L = 10 mH"
)

plt.xlabel("Time (s)")
plt.ylabel("Current (A)")

plt.grid()
plt.legend()



# -----------------------------
# PEAK CURRENT VS VOLTAGE
# -----------------------------

plt.figure(figsize=(8,5))


plt.plot(
    vamps[:len(peak_currents)],
    peak_currents,
    marker="o"
)


plt.title(
    "Peak current vs voltage amplitude\n"
    "Nonlinear inductor: L = 10 mH"
)

plt.xlabel("Voltage amplitude (V)")
plt.ylabel("Peak current (A)")

plt.grid()



# -----------------------------
# CURRENT ERROR CALCULATION
# -----------------------------

desired_current = (
    I_target_peak *
    np.sin(2*np.pi*f*best_t)
)


error = desired_current - best_i



plt.figure(figsize=(10,5))


plt.plot(
    best_t,
    desired_current,
    label="Desired current"
)

plt.plot(
    best_t,
    best_i,
    label="Actual current"
)

plt.title(
    "Desired vs actual current\n"
    "Highest voltage amplitude case"
)

plt.xlabel("Time (s)")
plt.ylabel("Current (A)")

plt.grid()
plt.legend()



plt.figure(figsize=(10,5))


plt.plot(
    best_t,
    error
)

plt.title(
    "Current waveform error\n"
    "Error = Desired - Actual"
)

plt.xlabel("Time (s)")
plt.ylabel("Error (A)")

plt.grid()



# -----------------------------
# SHOW ALL FIGURES
# -----------------------------

plt.show()