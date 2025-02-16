import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_and_clean_data(file_path):
    """
    Loads the CSV file, selects relevant columns, converts 'Price' to numeric, and assigns unique Test_IDs.
    """
    df = pd.read_csv(file_path)

    # Ensure we are selecting the correct columns
    df = df[['Offers', 'Price']].dropna(subset=['Offers']).reset_index(drop=True)

    # Convert 'Price' to numeric, forcing 'N/A' or other non-numeric values to NaN
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Create a unique ID for each test
    df['Test_ID'] = df.index + 1

    return df

def plot_data(df, file_name):
    """
    Plots the data with valid prices, missing values as red crosses, and offer markers.
    Saves the plot with an expressive filename.
    """
    # Identify missing price values
    missing_price = df[df['Price'].isna()]

    # Get unique offer values and their indices
    unique_offers = df['Offers'].unique()
    offer_indices = [df[df['Offers'] == offer].index[0] for offer in unique_offers]

    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(df['Test_ID'], df['Price'], marker='o', linestyle='-', label="Valid Data")

    # Mark missing values with a red cross
    plt.scatter(missing_price['Test_ID'], np.zeros(len(missing_price)), color='red', marker='x', s=100, label="Missing Price (N/A)")

    # Label the axes
    plt.xlabel("Test ID")
    plt.ylabel("Price")

    # Adjust xlim to align the first test ID with the Y-axis and remove extra space after the last test ID
    plt.xlim(df['Test_ID'].min(), df['Test_ID'].max())

    # Add vertical dashed lines and staggered text annotations for unique offers
    for i, offer in enumerate(unique_offers):
        idx = offer_indices[i]
        plt.axvline(x=df.loc[idx, 'Test_ID'], color='black', linestyle='dashed')

        y_offset = df['Price'].max(skipna=True)*1.008 if not df['Price'].isna().all() else 1
        x_offset = df.loc[idx, 'Test_ID'] + 9  # Move text slightly to the right
        plt.text(x_offset, y_offset, f"#off={offer}",
                 verticalalignment='bottom', horizontalalignment='center', fontsize=10, color='red')

    # Add a legend at the bottom
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Save the plot with an expressive filename
    output_file = f"price_optimality_{file_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_file}")

def process_all_files():
    """
    Reads all CSV files in the current directory matching 'generate_statistics_in_csv_*_by_offers.csv',
    processes them, and saves the plots.
    """
    csv_files = glob.glob("generate_statistics_in_csv_*_by_offers.csv")

    if not csv_files:
        print("No matching CSV files found.")
        return

    for file_path in csv_files:
        file_name = os.path.basename(file_path).replace("generate_statistics_in_csv_", "").replace("_by_offers.csv", "")
        print(f"Processing: {file_name}")

        # Load and clean data
        df = load_and_clean_data(file_path)

        # Plot and save output
        plot_data(df, file_name)

if __name__ == "__main__":
    process_all_files()
