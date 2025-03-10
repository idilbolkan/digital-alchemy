import os
import torch
import torchmetrics
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn

# Define paths
DATASET_PATH = "data/QM7X_Dataset/QM7X.db"
SAVE_DIR = "./ckpts"
MLFLOW_EXPERIMENT_NAME = "QuantumML-MolDynamics_QM7X"

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Define dataset properties
AVAILABLE_PROPERTIES = ["energy", "forces"]
CUTOFF_RADIUS   = 5.0
N_ATOM_BASIS    = 128
N_INTERACTIONS  = 6
BATCS_SIZE      = 128
LEARNING_RATE   = 1e-4

# Step 1: Load QM7-X dataset (unchanged)
qm7x_data = spk.data.AtomsDataModule(
    DATASET_PATH,
    batch_size=BATCS_SIZE,
    num_train=8000,
    num_val=1000,
    num_test=1000,
    transforms=[
        trn.ASENeighborList(cutoff=CUTOFF_RADIUS),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=4,
    pin_memory=True,
)

# Step 2: Define PaiNN Model (modified)
pairwise_distance = spk.atomistic.PairwiseDistances()
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=CUTOFF_RADIUS)

# schnet = spk.representation.SchNet(
#     n_atom_basis=N_ATOM_BASIS,
#     n_interactions=N_INTERACTIONS,
#     radial_basis=radial_basis,
#     cutoff_fn=spk.nn.CosineCutoff(CUTOFF_RADIUS),
# )

painn = spk.representation.PaiNN(
    n_atom_basis=N_ATOM_BASIS,
    n_interactions=N_INTERACTIONS,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(CUTOFF_RADIUS),
)

# Step 3: Define Output Modules (unchanged)
pred_energy = spk.atomistic.Atomwise(n_in=N_ATOM_BASIS, output_key="energy")
pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")

# Step 4: Assemble Neural Network Potential (modified representation)
nnpot = spk.model.NeuralNetworkPotential(
    representation=painn, #schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets("energy", add_mean=True, add_atomrefs=False)
    ]
)

# Step 5: Define Loss Functions (unchanged)
output_energy = spk.task.ModelOutput(
    name="energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()}
)

output_forces = spk.task.ModelOutput(
    name="forces",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()}
)

# Step 6: Define Training Task (unchanged)
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": LEARNING_RATE},
)

# Step 7: Integrate MLflow (modified representation name)
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run():
    mlflow.log_params({
        "representation": "painn",  # Changed from schnet
        "dataset": "QM7-X",
        "cutoff_radius": CUTOFF_RADIUS,
        "n_atom_basis": N_ATOM_BASIS,
        "n_interactions": N_INTERACTIONS,
        "batch_size": BATCS_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
    })

    # Define Trainer (unchanged)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(SAVE_DIR, "best_model_painn.ckpt"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )

    # Device setup (unchanged)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, training on CPU.")

    # Step 8: Train the Model (unchanged)
    trainer.fit(task, datamodule=qm7x_data)

    # Log model (modified name)
    mlflow.pytorch.log_model(task.model, "painn_model")  # Changed from schnet_model

    print("\nTraining complete! PaiNN model logged to MLflow.")
