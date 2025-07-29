import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ast

def create_experiment_group_key(df):
    """
    Cleans patient columns and creates a composite key for unique experiment groups.
    """
    def parse_and_sort(val):
        """Safely parses string representations of lists or single numbers."""
        try:
            # Handle NaN or other non-string types
            if pd.isna(val):
                return tuple()
            # Safely evaluate the string representation of the list
            if isinstance(val, str):
                # Handle space-separated numbers if not a list literal
                if '[' not in val:
                    parsed_list = [int(i) for i in val.split()]
                else:
                    parsed_list = ast.literal_eval(val)
            else:
                parsed_list = [val] # Handle single integer values

            if isinstance(parsed_list, int):
                 parsed_list = [parsed_list]

            return tuple(sorted(parsed_list))
        except (ValueError, SyntaxError):
            return tuple()

    # Apply the cleaning function to both columns
    df['patients_cleaned'] = df['patients'].apply(parse_and_sort)
    df['excluded_patients_cleaned'] = df['excluded_patients'].apply(parse_and_sort)
    
    # Create a composite key (a tuple of tuples) to uniquely identify the experiment
    df['experiment_group'] = list(zip(df['patients_cleaned'], df['excluded_patients_cleaned']))
    
    return df

# --- 1. Load and Prepare the Data ---
try:
    df = pd.read_csv('./output_metawears/results_noisy.csv')
except FileNotFoundError:
    print("Error: 'results_noisy.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Create the unique experimental group key
df = create_experiment_group_key(df)

# --- 2. Create the Reference (Baseline) DataFrame ---
ref_df = df[df['snr'] == 1000.0].copy()

# Create a lookup dictionary with the new, more specific key
baseline_auc_map = ref_df.set_index('experiment_group')['auc'].to_dict()

# --- 3. Calculate Performance Degradation ---
noisy_df = df[df['snr'] < 1000.0].copy()

# Map the baseline AUC using the composite experiment_group key
noisy_df['baseline_auc'] = noisy_df['experiment_group'].map(baseline_auc_map)

# Drop rows where there was no matching baseline
noisy_df.dropna(subset=['baseline_auc'], inplace=True)

# Calculate the performance degradation as a percentage
noisy_df['degradation'] = (noisy_df['baseline_auc'] - noisy_df['auc']) / noisy_df['baseline_auc']

# --- 4. Aggregate the Results ---
# Group by SNR and calculate the mean and std of the degradation across all experiments
degradation_stats = noisy_df.groupby('snr')['degradation'].agg(['mean', 'std']).reset_index()
degradation_stats = degradation_stats.sort_values('snr')

print("--- Performance Degradation Statistics (Grouped by [patients, excluded_patients]) ---")
print(degradation_stats)

# --- 5. Visualize the Results ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

plt.errorbar(
    degradation_stats['snr'],
    degradation_stats['mean'],
    yerr=degradation_stats['std'],
    fmt='-o',
    capsize=5,
    label='Mean Degradation',
    color='#0033A0', # A nice blue
    ecolor='#A0CBE8', # A lighter blue for error bars
    elinewidth=3,
    markersize=8
)

# Formatting the plot
plt.gca().invert_xaxis()
plt.title('Model Performance Degradation vs. SNR', fontsize=16, fontweight='bold')
plt.xlabel('Signal-to-Noise Ratio (SNR) in dB', fontsize=12)
plt.ylabel('Performance Degradation (Fractional AUC Loss)', fontsize=12)
plt.yticks(np.arange(0, degradation_stats['mean'].max() * 1.2, 0.05))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

# Show the plot
plt.show()