import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp


try:
    # Load the two CSV files into pandas DataFrames
    # Assumes the CSV files are in the same directory as the script
    proposed_df = pd.read_csv('./output_metawears/results_proposed.csv')
    maml_df = pd.read_csv('./output/results_MAML.csv')

    # Ensure the DataFrames have the same number of rows before proceeding
    if len(proposed_df) != len(maml_df):
        raise ValueError("The two CSV files do not have the same number of rows and cannot be merged row-by-row.")

    # Create a new DataFrame to store the comparison results
    # We select the 'patients' column from the proposed_df (assuming it's identical in both)
    # and the 'auc' columns from both, renaming them for clarity.
    comparison_df = pd.DataFrame({
        'Patients': proposed_df['patients'],
        'AUC (Proposed)': proposed_df['auc'],
        'AUC (MAML)': maml_df['auc']
    })

    # Calculate the difference between the two AUC scores
    # A positive value means the Proposed method's AUC was higher.
    # A negative value means the MAML method's AUC was higher.
    comparison_df['Difference (Proposed - MAML)'] = comparison_df['AUC (Proposed)'] - comparison_df['AUC (MAML)']
    comparison_df['Num Patients'] = comparison_df['Patients'].str.count(',') + 1


    # Display the final merged and calculated DataFrame
    # Using to_string() to ensure the full table is printed to the console
    print("Successfully merged the tables and calculated the difference:")
    print(comparison_df.to_string(index=False))

    print("Average of the differences:")
    print(comparison_df.groupby('Num Patients')['Difference (Proposed - MAML)'].mean())
    print("Standard deviation of the differences:")
    print(comparison_df.groupby('Num Patients')['Difference (Proposed - MAML)'].std())

    auc_improvement = comparison_df['Difference (Proposed - MAML)']

    # Perform a one-sample t-test
    t_statistic, p_value = ttest_1samp(a=auc_improvement, popmean=0)

    # Print the results
    print(f"Number of samples (experimental runs): {len(auc_improvement)}")
    print(f"Mean AUC Improvement: {auc_improvement.mean():.4f}")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create the box plot using seaborn
    sns.boxplot(
        # x='Num Patients',
        y='Difference (Proposed - MAML)',
        data=comparison_df,
        ax=ax,
        width=0.5,
        # Properties for the median line
        medianprops={'color': 'red', 'linewidth': 2},
        # Properties for the outlier points
        flierprops={'marker': 'o', 'markerfacecolor': 'skyblue', 'markeredgecolor': 'black', 'markersize': 8}
    )

    # Add a horizontal line at y=0 for reference
    ax.axhline(0, ls='--', color='gray', linewidth=1.5)

    # --- Set Titles and Labels ---
    ax.set_title(
        'Distribution of AUC Improvement (Proposed vs. MAML)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel(
        'Number of Patients in Target Set',
        fontsize=12,
        fontweight='bold'
    )
    ax.set_ylabel(
        'AUC Improvement (Proposed - MAML)',
        fontsize=12,
        fontweight='bold'
    )
    
    # Customize tick parameters for a cleaner look
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Ensure the layout is tight and display the plot
    plt.tight_layout()
    plt.show()




except FileNotFoundError as e:
    print(f"Error: Make sure the file '{e.filename}' is in the same directory as the script.")
except Exception as e:
    print(f"An error occurred: {e}")
