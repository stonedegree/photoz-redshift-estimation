import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def delta_z(z_pred, z_true):
    return (z_pred - z_true) / (1 + z_true)
    
def bias(z_pred, z_true):
    return np.mean(delta_z(z_pred, z_true))
    
def nmad(z_pred, z_true):    
    dz = delta_z(z_pred, z_true)    
    return 1.4826 * np.median(np.abs(dz - np.median(dz)))
    
def outlier_fraction(z_pred, z_true):
    dz = delta_z(z_pred, z_true)
    return np.mean(np.abs(dz) > 0.05)

def metrics(z_pred, z_true):
    print(f"Outlier Fraction: {outlier_fraction(z_pred, z_true):.16f}")
    print(f"Bias:             {bias(z_pred, z_true):.16f}")
    print(f"NMAD:             {nmad(z_pred, z_true):.16f}")

def plot_side_by_side_diagnostics(search, param_name, suptitle, xlabel_text, log=False):
    """Extract and plot the CV and Train scores side-by-side for a single hyperparameter."""
    
    results = pd.DataFrame(search.cv_results_)
    results[param_name] = results[f'param_{param_name}']

    # Group by the parameter to find the best performing score for each unique value
    best_idx = results.groupby(param_name)['mean_test_score'].idxmax()
    best_per_value = results.loc[best_idx].copy()

    # Sort the values numerically
    best_per_value = best_per_value.sort_values(param_name)

    # Create the values for x and y axis.
    x_values = best_per_value[param_name]
    cv_rmse = -best_per_value['mean_test_score']
    train_rmse = -best_per_value['mean_train_score']

    # Create figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left Plot: Cross-Validation
    axes[0].plot(x_values, cv_rmse, marker='o', color='red', linestyle='-', 
                 linewidth=2, label='Mean CV RMSE')
    axes[0].grid(True, linestyle='--', color='lightgray', alpha=0.8)
    axes[0].set_xlabel(xlabel_text, fontsize=12)
    axes[0].set_ylabel('RMSE (Lower is Better)', fontsize=12)
    axes[0].set_title('Cross-Validation Score', fontsize=14)
    axes[0].legend(loc='best', fontsize=11)
    
    # Force integer lables on x-axis
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Right Plot: Training
    axes[1].plot(x_values, train_rmse, marker='o', color='#1f77b4', linestyle='--', 
                 linewidth=2, label='Mean Training RMSE')
    axes[1].grid(True, linestyle='--', color='lightgray', alpha=0.8)
    axes[1].set_xlabel(xlabel_text, fontsize=12)
    axes[1].set_ylabel('RMSE (Lower is Better)', fontsize=12)
    axes[1].set_title('Training Score', fontsize=14)
    axes[1].legend(loc='best', fontsize=11)
    
    # Force integer lables on x-axis
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add a title for the whole figure
    fig.suptitle(suptitle, fontsize=16, fontweight='bold')

    # Turning the x-axis into log-scale
    if log:
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.show()

def optimize_hyper_params(base_model, X_tune, y_tune, param_grid, grid__random):
    total_runtime = 0
    cores = -4

    if(grid__random == 'random'):
        # Initialize RandomizedSearchCV object
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            scoring='neg_root_mean_squared_error',
            random_state=42,
            n_jobs=cores,
            n_iter=60,
            cv=3,
            verbose=1,
            return_train_score=True, # For plotting purposes
        )
    
        # Run the search
        print("-" * 5, "Running RandomizedSearchCV", "-" * 5)
        start_time = time.time()
        search.fit(X_tune, y_tune)
        end_time = time.time()
        
    else:
        # Initialize GridSearchCV object
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            n_jobs=cores,
            cv=3,
            verbose=1,
            return_train_score=True, # For plotting purposes
        )
    
        # Run the search
        print("-" * 5, "Running GridSearchCV", "-" * 5)
        start_time = time.time()
        search.fit(X_tune, y_tune)
        end_time = time.time()
    
    # Get total runtime (seconds)
    total_runtime = end_time - start_time
    unit = "seconds"
    
    # If runtime is more than 60 seconds, change units to minutes
    if(total_runtime > 60):
        total_runtime /= 60
        unit = "minutes"

    print(f"Total Runtime: {total_runtime:.4f} {unit}")
        
    # Return the search (for plot), best hyperparameters, and best rmse
    return (search, search.best_params_, search.best_score_)

def binned_metrics(z_pred, z_true, bin_var, bin_edges, label):
    print("-" * 4, label, "-" * 4)
    print(f"{'bin':<12}{'N':<10}{'bias':<25}{'nmad':<25}{'outlier_frac':<25}")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        
        # 1. Create the mask using the new bin_var (e.g., i_mag_val)
        mask = (bin_var >= lo) & (bin_var < hi)
        n = mask.sum()
        
        if n == 0:
            continue
            
        # 2. Extract the actual redshifts for the math
        zt, zp = z_true[mask], z_pred[mask]
        
        print(f"{lo:.2f}-{hi:.2f}   {n:<10}{bias(zp, zt):<25.16f}{nmad(zp, zt):<25.16f}{outlier_fraction(zp, zt):<25.16f}")
    print("\n")

def plot_binned_metric(feature_array, bins, xlabel, ylabel, title, metric_type='nmad', z_pred=None, z_true=None, width_array=None):
    import matplotlib.pyplot as plt
    
    bin_centers = []
    y_values = []
    
    for i in range(len(bins)-1):
        mask = (feature_array >= bins[i]) & (feature_array < bins[i+1])
        
        # Only plot bins that have a statistically useful number of galaxies
        if mask.sum() > 10:
            bin_centers.append(f"{bins[i]:.4f}-{bins[i+1]:.4f}")
            
            # Calculate the appropriate metric for the y-axis
            if metric_type == 'nmad':
                y_values.append(nmad(z_pred[mask], z_true[mask]))
            elif metric_type == 'width':
                y_values.append(width_array[mask].mean())
                
    plt.figure(figsize=(7, 4))
    
    # Navy for accuracy, maroon for uncertainty
    line_color = 'navy' if metric_type == 'nmad' else 'maroon'
    
    plt.plot(bin_centers, y_values, marker='o', color=line_color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()