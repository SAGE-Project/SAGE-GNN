import os
import json
import re
import numpy as np
from collections import defaultdict

def compute_gini_coefficient(values):
    """
    Compute the Gini coefficient from a list of values.
    """
    values = np.array(values)
    print("values ", values)
    values_sorted = np.sort(values)  # Sort the values
    n = len(values)
    cumulative_values = np.cumsum(values_sorted)
    gini = (1 / n) * (n + 1 - 2 * np.sum(cumulative_values) / cumulative_values[-1])
    return gini

def find_min_price(data):
    """
    Recursively search for 'min_price' in nested data and yield its values.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "min_price":
                yield value
            else:
                yield from find_min_price(value)
    elif isinstance(data, list):
        for item in data:
            yield from find_min_price(item)

def extract_file_id(file_name):
    """
    Extract the numeric ID from a file name (e.g., 'WordPress3_9102.json' -> 9102).
    """
    match = re.search(r"(\d+)", file_name)
    return int(match.group(1)) if match else None

def count_min_price_and_files(directory):
    """
    Count occurrences of 'min_price' values and track the files where each minimum value occurs.
    """
    min_price_counts = defaultdict(int)  # To count occurrences of each value
    min_price_files = defaultdict(set)  # To track files where each value is minimum

    # Iterate through the files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)

                    # Find all 'min_price' values in the data
                    min_prices = list(find_min_price(data))
                    if not min_prices:
                        continue  # Skip files with no 'min_price'

                    # Determine the minimum value in the file
                    min_value = min(min_prices)

                    # Increment the count for all occurrences of 'min_price'
                    for value in min_prices:
                        min_price_counts[value] += 1

                    # Track the file for the minimum value
                    file_id = extract_file_id(file_name)
                    if file_id is not None:
                        min_price_files[min_value].add(file_name)

            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {file_name}: {e}")

    # Convert counts dictionary to a list of tuples [(count, value), ...]
    occurrences = [(count, value) for value, count in min_price_counts.items()]

    # Convert file tracking to a list of tuples [(value, {files}), ...]
    files_with_min_values = [(value, sorted(files)) for value, files in min_price_files.items()]

    # Compute Gini coefficient
    gini_coefficient = compute_gini_coefficient(list(min_price_counts.values())) if min_price_counts else 0

    return occurrences, files_with_min_values, gini_coefficient

def main():
    # Define input and output directories
    directory_path = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Datasets/DsSecureWebContainer_40_36"
    output_file = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Output/Stats-Gini/out-DsSecureWebContainer_40_36.txt"  # Change this to your desired output path

    # Validate input directory
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    # Process files and calculate results
    occurrences, files_with_min_values, gini = count_min_price_and_files(directory_path)

    # Write results as arrays of tuples
    with open(output_file, "w") as file:
        file.write("# Results in the form of arrays of tuples\n\n")
        file.write("occurrences (min_val, number of occurences) = ")
        file.write(repr(sorted(occurrences, reverse=True)) + "\n\n")

        file.write("diversity of labels: len(occurrences)= ")
        file.write(repr(len(occurrences)) + "\n\n")

        file.write("files_with_min_values = ")
        file.write(repr(sorted(files_with_min_values)) + "\n\n")

        file.write("gini_coefficient = ")
        file.write(repr(gini) + "\n")

    print("Results written to file successfully.")

# Entry point
if __name__ == "__main__":
    main()
