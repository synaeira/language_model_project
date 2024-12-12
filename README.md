# Transformer-Based Text Generation Project

This project implements a custom Transformer model to perform text generation tasks. The task is detailed in task.ipynb as part of the Master 2 Deep Learning course. 

The structure is modular, with components for dataset processing, model training, and testing.

---

## Project Structure

### Main Folders and Files

- **datasets/**: Contains the text datasets used for training and testing.
  - `jul-data.txt`, `shakespeare-data.txt`: Full datasets.
  - `jul-data-subset.txt`, `shakespeare-data-subset.txt`: Smaller datasets for quick experimentation.
  - `subset.py`: Script to create reduced datasets from full datasets.

- **src/**: Contains all the source code for the project.
  - **model/**: Includes the core building blocks of the Transformer model:
    - `embedding.py`: Implements token embedding for input processing.
    - `mha.py`: Multi-head attention layer.
    - `mlp.py`: Feed-forward network layer.
    - `transformer_block.py`: Defines a single Transformer block.
    - `transformer.py`: Assembles the full Transformer model.
  - `data_processing.py`: Processes the datasets to:
    - Create a dictionary (vocabulary) of unique characters.
    - Split the text into overlapping chunks for training.
  - `training.py`: Contains the `Trainer` class for training the Transformer.
  - `testing.py`: Includes a function to generate text from a trained model.
  - `results.ipynb`: Example notebook demonstrating:
    - Dataset usage (e.g., Shakespeare and Jul datasets).
    - Training with the `Trainer` class.
    - Text generation using the trained Transformer.

- **requirements.txt**: Lists all Python dependencies required for the project.
- **README.md**: Current documentation.
- **LICENSE**: License for the project.
- **task.ipynb**: Detailled task of the project.

---

## Setup

All required dependencies are listed in `requirements.txt`. To ensure a smooth setup, install them using:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch`: For building and training the Transformer.
- `torchvision`: Utility functions for PyTorch.
- `tqdm`: For progress bars.
- `wandb`: For experiment tracking (optional).

---

## Usage

### Training a Transformer Model

1. Instantiate the `Trainer` class with the required parameters:

   ```python
   from training import Trainer

   train = Trainer(
       datafile="../datasets/shakespeare-data.txt",
       block_size=128,
       batch_size=128,
       dim_emb=768,
       hidden_layer=128,
       num_head=8,
       num_transformer=12,
       learning_rate=0.0001,
       iteration=6000
   )
   ```

2. Start training:
   ```python
   train.run()
   ```

3. Save the trained model:
   ```python
   train.save_model(path="../trained_models/shakespeare-model.pth")
   ```

### Loading a Pre-Trained Model

1. Load the model:

   ```python
   train = Trainer(load_path="../trained_models/shakespeare-model.pth")
   ```

2. If the parameters of the loaded model differ from the defaults, update them as needed.

### Generating Text

Use the `testing.py` module to generate text:

```python
from testing import generate_text

initial_phrase = "O God, O God!"
output = generate_text(
    model=train.model,
    initial_phrase=initial_phrase,
    num_chars=1000,
    block_size=128
)
print(output)
```

### Example Workflow

Refer to `results.ipynb` for a complete example:
- Illustrates training on the Shakespeare and Jul datasets.
- Demonstrates text generation with the trained model.
- Provides insights into how to use the datasets and processing tools.

---

## Dataset Details

### Available Datasets
- **Shakespeare Dataset**: A collection of text from William Shakespeare's works.
- **Jul Dataset**: A collection of lyrics by Jul (french rapper).

### Subset Creation
Use `subset.py` to generate smaller datasets for experimentation. Adjust parameters to specify the desired subset size.

---

## Authors
- Charles MARTIN (https://github.com/Charles205050)
- Stacy DEMANGE (https://github.com/synaeira)