# Simple CNN on CIFAR-10

## Description
This project implements a simple Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR-10 dataset. It includes a basic training pipeline, three experiment configurations and evaluation using accuracy and loss.

## Installation (uv)
Clone the repository and install dependencies using uv:

```bash
git clone <your-repo-url>
cd <your-project>
uv sync
```

Requirements:
- Python 3.8+
- uv installed

Install uv if needed:
```bash
pip install uv
```
## How to Run

From the `src/` directory, run:
```bash
python main.py
```

This will:
1. Load CIFAR-10
2. Train the model for each experiment
3. Evaluate performance
4. Print results in a table

## Project structure

```
.
├── data/            # CIFAR-10 dataset (DVC tracked)
├── src/
│   ├── config.py    # Experiment configurations
│   ├── dataset.py   # Data loading & preprocessing
│   ├── model.py     # CNN architecture
│   ├── train.py     # Training loop
│   └── main.py      # Entry point
│
├── eda.ipynb        # Exploratory data analysis
├── results.png      # Example results
├── pyproject.toml   # Dependencies
└── README.md
```

## Dataset

This project uses the CIFAR-10 dataset, a widely used benchmark for image classification tasks.
It consists of 60,000 color images of size 32×32 pixels, divided into 10 classes.

The dataset is split into:

* 50,000 training images
* 10,000 test images

Images are automatically downloaded and stored in the `data/` directory.


## Data Preprocessing and Augmentation
The training data is augmented using random transformations to improve generalization.
The following transformations are applied:

* Random horizontal flip (p=0.5)
* Random crop (32x32) with padding=4
* Conversion to tensor
* Normalization using dataset-specific mean and standard deviation

Augmentation is applied on-the-fly during training, but not during evaluation. 

## Model

The model is a simple Convolutional Neural Network (CNN) consisting of:

- 3 convolutional layers
- ReLU activations
- Max pooling layers
- Dropout for regularization
- 2 fully connected layers

**Input:** 32x32 RGB images
**Output:** 10 classes

## Experiments

The project includes a set of predefined experiments to compare different training configurations. 
Each experiment varies hyperparameters such as learning rate, number of epochs, and batch size.

```python
EXPERIMENTS = [
    {
        "name": "baseline",
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 64,
    },
    {
        "name": "low_lr",
        "epochs": 15,
        "lr": 0.0003,
        "batch_size": 64,
    },
    {
        "name": "more_epochs",
        "epochs": 20,
        "lr": 0.001,
        "batch_size": 128,
    },
]
```

**baseline**: Default setup with a standard learning rate, 0.001, and a moderate training time.

**low_lr**: Uses a lower learning rate, 0.0003, and more epochs, allowing for more gradual learning.

**more_epochs**: Trains longer with a larger batch size (128), which may improve stability and performance.

## Experiment Output
Final results:

===== RESULTS =====
| Experiment   | Train Loss | Test Loss | Accuracy |
|-------------|-----------|----------|----------|
| baseline    | 0.7016    | 0.6329   | 0.7815   |
| low_lr      | 0.6882    | 0.6451   | 0.7726   |
| more_epochs | 0.5613    | 0.5754   | 0.7999   |

## Notes
The dataset is downloaded automatically to the data/ folder. 
GPU will be used if available. 
Dependencies are defined in pyproject.toml and locked in uv.lock
uv sync automatically creates and manages a virtual environment.

## Acknowledgments
This project was developed with assistance from ChatGPT, which was used to help generate and refine parts of the Python code.

