from QSPICEBatchRunner import QSPICEBatchRunner
import numpy as np

# Create runner
runner = QSPICEBatchRunner(
    basefile="Test",
    workdir="L_batch_4"
)

# Convert schematic to .cir
runner.qsch_to_cir(r"C:\Users\aycah\Documents\RISE\qspice_and_training\Test.qsch")

# Create sweep values
param_list = [
    {"LVAL": lval}
    for lval in np.linspace(1e-3,100e-3)
]

# Generate modified netlists
cir_files = runner.generate_param_cir_files(param_list)

# Run simulations
results = runner.run_batch(
    cir_files,
    signals=["I(L1)"],
    max_workers=4
)

# Plot all currents
runner.plot_sweep(
    results,
    signal_name="I(L1)",
    xlabel="Time (s)",
    ylabel="Current (A)",
    title="Nonlinear Inductor Inductance Sweep"
)