import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numba import jit
import argparse
import os
import time
from scipy.sparse import csr_matrix
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class TravelTimeEstimator:
    """
    A class to estimate travel times using Matrix Factorization with ALS.
    """
    
    def __init__(self, rank=10, lambda_=0.1, iterations=100, verbose=True):
        """
        Initialize the model parameters.
        
        Parameters:
        -----------
        rank : int
            Number of latent factors
        lambda_ : float
            Regularization parameter
        iterations : int
            Maximum number of iterations for ALS
        verbose : bool
            Whether to print progress during training
        """
        self.rank = rank
        self.lambda_ = lambda_
        self.iterations = iterations
        self.verbose = verbose
        self.U = None  # Road segment features
        self.V = None  # Time interval features
        self.error_history = []
        
    def fit(self, R, mask=None):
        """
        Fit the model using Alternating Least Squares.
        
        Parameters:
        -----------
        R : numpy.ndarray
            Travel time matrix (road segments × time intervals)
        mask : numpy.ndarray, optional
            Binary mask indicating observed entries (1 = observed)
            
        Returns:
        --------
        self : TravelTimeEstimator
            The fitted model
        """
        m, n = R.shape
        
        # Initialize factor matrices
        self.U = np.random.rand(m, self.rank)
        self.V = np.random.rand(n, self.rank)
        
        # If no mask is provided, use non-NaN values
        if mask is None:
            mask = ~np.isnan(R)
        
        # Fill NaN values with zeros for computation
        R_filled = np.copy(R)
        R_filled[np.isnan(R_filled)] = 0
        
        # ALS optimization
        for iteration in range(self.iterations):
            # Update U (road segments)
            for i in range(m):
                # Indices of observed time intervals for this road segment
                idx = np.where(mask[i, :])[0]
                if len(idx) > 0:
                    V_i = self.V[idx, :]
                    R_i = R_filled[i, idx]
                    
                    # Regularized least squares solution
                    A = V_i.T @ V_i + self.lambda_ * np.eye(self.rank)
                    b = V_i.T @ R_i
                    self.U[i, :] = np.linalg.solve(A, b)
            
            # Update V (time intervals)
            for j in range(n):
                # Indices of observed road segments for this time interval
                idx = np.where(mask[:, j])[0]
                if len(idx) > 0:
                    U_j = self.U[idx, :]
                    R_j = R_filled[idx, j]
                    
                    # Regularized least squares solution
                    A = U_j.T @ U_j + self.lambda_ * np.eye(self.rank)
                    b = U_j.T @ R_j
                    self.V[j, :] = np.linalg.solve(A, b)
            
            # Compute current error on observed entries
            pred = self.U @ self.V.T
            error = np.sqrt(np.sum((R_filled[mask] - pred[mask])**2) / np.sum(mask))
            self.error_history.append(error)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.iterations}, RMSE: {error:.4f}")
                
            # Early stopping if error is small enough
            if error < 1e-4:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
                
        return self
    
    def predict(self):
        """
        Generate predictions using the trained model.
        
        Returns:
        --------
        numpy.ndarray
            Reconstructed travel time matrix
        """
        if self.U is None or self.V is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        return self.U @ self.V.T
    
    def predict_for_segment_time(self, segment_id, time_id):
        """
        Predict travel time for a specific road segment and time interval.
        
        Parameters:
        -----------
        segment_id : int
            Index of the road segment
        time_id : int
            Index of the time interval
            
        Returns:
        --------
        float
            Predicted travel time
        """
        if self.U is None or self.V is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        return np.dot(self.U[segment_id, :], self.V[time_id, :])
    
    def plot_convergence(self, save_path=None):
        """
        Plot convergence of the ALS algorithm.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.error_history) + 1), self.error_history, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('Convergence of ALS Algorithm')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        

class DataGenerator:
    """
    A class to generate synthetic travel time data for testing.
    """
    
    @staticmethod
    def generate_synthetic_data(n_segments=100, n_times=48, sparsity=0.7, noise_level=0.1):
        """
        Generate synthetic travel time data.
        
        Parameters:
        -----------
        n_segments : int
            Number of road segments
        n_times : int
            Number of time intervals
        sparsity : float
            Fraction of missing values (0 to 1)
        noise_level : float
            Standard deviation of Gaussian noise
            
        Returns:
        --------
        tuple
            (true_matrix, observed_matrix, mask)
        """
        # Generate road segment features (e.g., length, capacity)
        segment_features = np.random.rand(n_segments, 3)
        
        # Generate time interval features (e.g., time of day patterns)
        time_features = np.zeros((n_times, 3))
        
        # Create time of day patterns (morning peak, afternoon peak)
        for i in range(n_times):
            # Morning peak around interval 16-20 (8-10 AM)
            morning_effect = np.exp(-0.1 * (i - 18)**2) if i <= 24 else 0
            # Evening peak around interval 34-38 (5-7 PM)
            evening_effect = np.exp(-0.1 * (i - 36)**2) if i > 24 else 0
            # Weekend effect (lower travel times)
            weekend_effect = 0.3 if i % 7 in [0, 6] else 0
            
            time_features[i] = [morning_effect, evening_effect, weekend_effect]
        
        # Generate true travel time matrix based on features
        true_matrix = np.zeros((n_segments, n_times))
        for i in range(n_segments):
            for j in range(n_times):
                # Base travel time for this segment
                base_time = 2 + 8 * segment_features[i, 0]
                
                # Adjust for time of day effects
                time_factor = 1.0 + 0.5 * time_features[j, 0] + 0.7 * time_features[j, 1]
                
                # Weekend discount
                if time_features[j, 2] > 0:
                    time_factor *= 0.8
                    
                true_matrix[i, j] = base_time * time_factor
        
        # Add some noise to simulate real-world variations
        true_matrix += np.random.normal(0, noise_level * np.mean(true_matrix), true_matrix.shape)
        true_matrix = np.maximum(true_matrix, 0.1)  # Ensure positive travel times
        
        # Create mask for observed entries
        mask = np.random.rand(n_segments, n_times) > sparsity
        
        # Create observed matrix with missing values
        observed_matrix = np.copy(true_matrix)
        observed_matrix[~mask] = np.nan
        
        return true_matrix, observed_matrix, mask
    
    @staticmethod
    def create_dataframe_from_matrix(matrix, mask=None):
        """
        Convert a travel time matrix to a DataFrame format.
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            Travel time matrix (road segments × time intervals)
        mask : numpy.ndarray, optional
            Binary mask indicating observed entries
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with road segment, time interval, and travel time
        """
        n_segments, n_times = matrix.shape
        
        # Create empty lists to store data
        segment_ids = []
        time_ids = []
        travel_times = []
        
        # Generate start date and time intervals
        start_date = datetime(2025, 5, 9)  # Today's date
        time_intervals = [start_date + timedelta(minutes=30*i) for i in range(n_times)]
        
        # Fill the lists
        for i in range(n_segments):
            for j in range(n_times):
                if mask is None or mask[i, j]:
                    segment_ids.append(f"Segment_{i+1}")
                    time_ids.append(time_intervals[j])
                    travel_times.append(matrix[i, j])
        
        # Create DataFrame
        df = pd.DataFrame({
            'segment_id': segment_ids,
            'timestamp': time_ids,
            'travel_time': travel_times
        })
        
        return df


class Visualizer:
    """
    A class to visualize travel time data and model results.
    """
    
    @staticmethod
    def plot_heatmap(matrix, title, save_path=None):
        """
        Plot a heatmap of a travel time matrix.
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            Travel time matrix to visualize
        title : str
            Title of the plot
        save_path : str, optional
            Path to save the figure
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, cmap='viridis', robust=True)
        plt.title(title)
        plt.xlabel('Time Interval')
        plt.ylabel('Road Segment')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_comparison(true_matrix, observed_matrix, predicted_matrix, segment_id=0, save_path=None):
        """
        Plot a comparison of true, observed, and predicted travel times for a road segment.
        
        Parameters:
        -----------
        true_matrix : numpy.ndarray
            True travel time matrix
        observed_matrix : numpy.ndarray
            Observed travel time matrix with missing values
        predicted_matrix : numpy.ndarray
            Predicted travel time matrix
        segment_id : int
            Road segment to visualize
        save_path : str, optional
            Path to save the figure
        """
        plt.figure(figsize=(14, 7))
        
        x = np.arange(true_matrix.shape[1])
        
        # True values
        plt.plot(x, true_matrix[segment_id], 'b-', label='True')
        
        # Observed values (non-NaN)
        mask = ~np.isnan(observed_matrix[segment_id])
        plt.scatter(x[mask], observed_matrix[segment_id, mask], color='green', marker='o', label='Observed')
        
        # Predicted values
        plt.plot(x, predicted_matrix[segment_id], 'r--', label='Predicted')
        
        plt.xlabel('Time Interval')
        plt.ylabel('Travel Time (minutes)')
        plt.title(f'Travel Time Comparison for Road Segment {segment_id+1}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_interactive_time_series(df, segment_ids=None, save_path=None):
        """
        Create an interactive time series plot using Plotly.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with travel time data
        segment_ids : list, optional
            List of segment IDs to visualize
        save_path : str, optional
            Path to save the figure
        """
        if segment_ids is None:
            segment_ids = df['segment_id'].unique()[:5]  # Show first 5 segments by default
        
        # Filter data for selected segments
        df_filtered = df[df['segment_id'].isin(segment_ids)]
        
        # Create figure
        fig = px.line(df_filtered, x='timestamp', y='travel_time', color='segment_id',
                      title='Real-time Travel Time Trends',
                      labels={'travel_time': 'Travel Time (minutes)', 'timestamp': 'Time'})
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Travel Time (minutes)',
            legend_title='Road Segment',
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        fig.show()


def evaluate_model(true_matrix, predicted_matrix, mask=None):
    """
    Evaluate model performance using various metrics.
    
    Parameters:
    -----------
    true_matrix : numpy.ndarray
        True travel time matrix
    predicted_matrix : numpy.ndarray
        Predicted travel time matrix
    mask : numpy.ndarray, optional
        Binary mask indicating observed entries
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    if mask is None:
        mask = ~np.isnan(true_matrix)
    
    # Extract valid entries
    y_true = true_matrix[mask]
    y_pred = predicted_matrix[mask]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    }


def run_experiment(n_segments=100, n_times=48, rank=10, lambda_=0.1, 
                  iterations=100, sparsity=0.7, output_dir='results'):
    """
    Run a complete experiment to test the matrix factorization approach.
    
    Parameters:
    -----------
    n_segments : int
        Number of road segments
    n_times : int
        Number of time intervals
    rank : int
        Number of latent factors
    lambda_ : float
        Regularization parameter
    iterations : int
        Maximum number of iterations for ALS
    sparsity : float
        Fraction of missing values (0 to 1)
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics and timing information
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Generate synthetic data
    print("Generating synthetic data...")
    true_matrix, observed_matrix, mask = DataGenerator.generate_synthetic_data(
        n_segments=n_segments, n_times=n_times, sparsity=sparsity
    )
    
    # Step 2: Train model
    print("Training model...")
    start_time = time.time()
    model = TravelTimeEstimator(rank=rank, lambda_=lambda_, iterations=iterations)
    model.fit(observed_matrix, mask)
    training_time = time.time() - start_time
    
    # Step 3: Generate predictions
    predicted_matrix = model.predict()
    
    # Step 4: Evaluate model
    metrics = evaluate_model(true_matrix, predicted_matrix, mask)
    metrics['training_time'] = training_time
    
    # Step 5: Create visualizations
    print("Creating visualizations...")
    
    # Heatmaps
    Visualizer.plot_heatmap(true_matrix, 'True Travel Time Matrix', 
                          os.path.join(output_dir, 'true_heatmap.png'))
    
    Visualizer.plot_heatmap(observed_matrix, 'Observed Travel Time Matrix (with missing values)', 
                          os.path.join(output_dir, 'observed_heatmap.png'))
    
    Visualizer.plot_heatmap(predicted_matrix, 'Predicted Travel Time Matrix', 
                          os.path.join(output_dir, 'predicted_heatmap.png'))
    
    # Convergence plot
    model.plot_convergence(os.path.join(output_dir, 'convergence.png'))
    
    # Compare true vs. predicted for a few segments
    for i in range(3):  # First 3 segments
        Visualizer.plot_comparison(true_matrix, observed_matrix, predicted_matrix, i, 
                                os.path.join(output_dir, f'comparison_segment_{i+1}.png'))
    
    # Create interactive time series
    df_true = DataGenerator.create_dataframe_from_matrix(true_matrix)
    df_pred = DataGenerator.create_dataframe_from_matrix(predicted_matrix)
    df_pred['type'] = 'Predicted'
    df_true['type'] = 'True'
    df_combined = pd.concat([df_true, df_pred])
    
    Visualizer.plot_interactive_time_series(
        df_combined, segment_ids=[f'Segment_{i+1}' for i in range(3)],
        save_path=os.path.join(output_dir, 'interactive_time_series.html')
    )
    
    # Print results
    print("\nExperiment Results:")
    print(f"Number of road segments: {n_segments}")
    print(f"Number of time intervals: {n_times}")
    print(f"Matrix sparsity: {sparsity:.2f}")
    print(f"Rank: {rank}")
    print(f"Lambda: {lambda_}")
    print(f"Training time: {training_time:.2f} seconds")
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if metric != 'training_time':
            print(f"{metric}: {value:.4f}")
    
    return metrics


def main():
    """
    Main function to parse command line arguments and run the experiment.
    """
    parser = argparse.ArgumentParser(description='Real-time Travel Time Estimation Using Matrix Factorization')
    
    parser.add_argument('--segments', type=int, default=100,
                        help='Number of road segments (default: 100)')
    parser.add_argument('--times', type=int, default=48,
                        help='Number of time intervals (default: 48)')
    parser.add_argument('--rank', type=int, default=10,
                        help='Number of latent factors (default: 10)')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.1,
                        help='Regularization parameter (default: 0.1)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Maximum number of iterations (default: 100)')
    parser.add_argument('--sparsity', type=float, default=0.7,
                        help='Fraction of missing values (default: 0.7)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    # Run experiment with specified parameters
    run_experiment(
        n_segments=args.segments,
        n_times=args.times,
        rank=args.rank,
        lambda_=args.lambda_,
        iterations=args.iterations,
        sparsity=args.sparsity,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()