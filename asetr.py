from ase.io import Trajectory
traj = Trajectory("simulation/trajectory.traj")
for step, atoms in enumerate(traj):
    print(f"Step {step}: Energy = {atoms.get_potential_energy()} eV")