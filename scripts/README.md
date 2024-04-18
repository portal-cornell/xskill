### Training

1. Run the skill discovery script (visual representation alignment module):
   ```bash
   python scripts/skill_discovery.py
   ```

2. Compute latent embeddings for all human and robot videos offline:
   ```bash
   python scripts/label_sim_kitchen_dataset.py
   ```

3. Run the skill_transfer_composing script (skill alignment transformer + policy):
   ```bash
   python scripts/skill_transfer_composing.py
   ```

### Evaluation

1. Evaluate Policy:
   ```bash
   python scripts/eval_checkpoint.py
   ```

2. Compute nearest neighbors/temporal cyclic consistency metrics:
   ```bash
   python scripts/compute_nearest_neighbors.py
   ```

3. Plot top most activated prototypes per subtask:
   ```bash
   python scripts/plot_protos.py
   ```

### Data Processing

1. Generate l2 distance matrices for each robot video with respect to all human z's:
   ```bash
   python scripts/generate_human_data_bank.py
   ```

