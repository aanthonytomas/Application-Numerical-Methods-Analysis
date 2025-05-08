# Real-Time Travel Time Estimation Using Matrix Factorization

This project provides a comprehensive Python-based implementation for estimating real-time travel times using matrix factorization techniques, specifically Alternating Least Squares (ALS). It includes tools for generating synthetic travel time data, training a model, evaluating its performance, and visualizing results through various plots and interactive graphs.

## Features

### 1. **Travel Time Estimation**
- Implements a matrix factorization approach to predict travel times for road segments over specified time intervals.
- Supports missing data handling with customizable sparsity levels.

### 2. **Synthetic Data Generation**
- Generates synthetic travel time data for testing purposes, simulating road segment and time interval features.
- Adds Gaussian noise to emulate real-world variations.

### 3. **Evaluation Metrics**
- Calculates key evaluation metrics such as:
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**
  - **Mean Absolute Percentage Error (MAPE)**
  - **R-squared (R²)**

### 4. **Visualization**
- Provides rich visualizations for analyzing results:
  - Heatmaps for true, observed, and predicted travel time matrices.
  - Convergence plots to track ALS optimization progress.
  - Segment-wise comparisons of true, observed, and predicted travel times.
  - Interactive time-series plots using Plotly.

### 5. **Command-Line Interface**
- Fully configurable via command-line arguments for parameters such as:
  - Number of road segments and time intervals.
  - Rank of latent factors, regularization parameter, and maximum iterations.
  - Fraction of missing values and output directory.

### 6. **Reproducibility**
- Random seeds are set for reproducibility of results.

---

## How It Works

### 1. **TravelTimeEstimator**
A class that implements matrix factorization with ALS:
- **Fit**: Trains the model on an observed travel time matrix with missing values.
- **Predict**: Reconstructs the full matrix, including missing entries.
- **Convergence Plot**: Visualizes the RMSE over iterations to show model training progress.

### 2. **DataGenerator**
A utility class to generate synthetic travel time data:
- Simulates road segment and time features with time-of-day and weekend effects.
- Outputs the true matrix, observed matrix (with missing values), and a binary mask.

### 3. **Visualizer**
A class for creating visualizations:
- Heatmaps of travel time matrices.
- Segment-wise comparison plots.
- Interactive time-series plots for real-time trends.

### 4. **Evaluation**
Calculates performance metrics to assess model accuracy on the observed data.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/travel-time-estimation.git
   cd travel-time-estimation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Command-Line Interface
Run the script with optional arguments to configure the experiment:
```bash
python main.py --segments 100 --times 48 --rank 10 --lambda 0.1 --iterations 100 --sparsity 0.7 --output results
```

### Example Parameters
- `--segments`: Number of road segments (default: 100).
- `--times`: Number of time intervals (default: 48).
- `--rank`: Number of latent factors (default: 10).
- `--lambda`: Regularization parameter (default: 0.1).
- `--iterations`: Maximum number of iterations for ALS (default: 100).
- `--sparsity`: Fraction of missing values (default: 0.7).
- `--output`: Output directory for results (default: `results`).

---

## Results

After running the experiment, the following outputs are generated:
1. **Heatmaps**: Visualize true, observed, and predicted travel times.
2. **Convergence Plot**: Tracks RMSE over ALS iterations.
3. **Comparison Plots**: Compare true, observed, and predicted values for specific road segments.
4. **Interactive Graphs**: Explore trends in travel times across different segments.

---

## Example Workflow

### Step 1: Generate Synthetic Data
```python
from DataGenerator import DataGenerator
true_matrix, observed_matrix, mask = DataGenerator.generate_synthetic_data()
```

### Step 2: Train the Model
```python
from TravelTimeEstimator import TravelTimeEstimator
model = TravelTimeEstimator(rank=10, lambda_=0.1, iterations=100)
model.fit(observed_matrix, mask)
predicted_matrix = model.predict()
```

### Step 3: Evaluate Performance
```python
from evaluate_model import evaluate_model
metrics = evaluate_model(true_matrix, predicted_matrix, mask)
print(metrics)
```

### Step 4: Visualize Results
```python
from Visualizer import Visualizer
Visualizer.plot_heatmap(predicted_matrix, "Predicted Travel Time Matrix")
```

---

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
- `numba`
- `argparse`

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
Special thanks to the open-source community for providing the tools and libraries needed to build this project.