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
iterations = 50

dt = 1e-5
t = np.arange(0, 1/f, dt)


# desired current

I_target = 1

i_reference = (
    I_target *
    np.sin(2*np.pi*f*t)
)


# initial voltage guess

V_amp = 10

voltage = (
    V_amp *
    np.sin(2*np.pi*f*t)
)


learning_rate = 0.5


# store history

error_history = []
voltage_history = []
current_history = []



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


    print(
        "RMS error:",
        np.sqrt(np.mean(error**2))
    )


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



# -----------------------------------
# PLOTS
# -----------------------------------

plt.figure(figsize=(12,5))


for k in range(iterations):

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


for k in range(iterations):

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


for k in range(iterations):

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

# RMS ERROR VS ITERATION
rms_per_iteration = [
    np.sqrt(np.mean(e**2))
    for e in error_history
]

plt.figure(figsize=(8, 4))
plt.plot(
    range(1, iterations + 1),
    rms_per_iteration,
    marker='o'
)
plt.title("RMS Error vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("RMS Error (A)")
plt.grid()
plt.tight_layout()


plt.show()