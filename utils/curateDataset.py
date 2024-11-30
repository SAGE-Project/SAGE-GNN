import os
import json


def delete_files_without_matching_min_price(directory, min_price_array):
    """
    Deletes files in the directory if their 'min_price' (in 'output' field) does NOT match
    any of the 'y' values in the provided array [(x1, y1), (x2, y2), ...].

    Also counts and prints the first arguments (x) in the min_price_array.

    Args:
        directory (str): Path to the directory.
        min_price_array (list of tuples): List of tuples where the second value is 'min_price'.
    """
    files = sorted(os.listdir(directory))  # Sort files to ensure order
    deleted_count = 0  # Track the number of deleted files
    x_values_count = {}  # Dictionary to count occurrences of the first argument (x)

    # Create a set of y values for fast matching
    y_values_set = set(y for _, y in min_price_array)

    # Count occurrences of x values in min_price_array
    for x, y in min_price_array:
        if x not in x_values_count:
            x_values_count[x] = 0
        x_values_count[x] += 1

    print(f"Count of x values from min_price_array: {x_values_count}")

    for filename in files:
        filepath = os.path.join(directory, filename)

        # Skip if it's not a file
        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, 'r') as file:
                content = file.read()

                # Parse the file as JSON
                try:
                    data = json.loads(content)
                    print(f"Parsed data from {filename}: {data}")  # Debug: Check parsed data
                except json.JSONDecodeError as e:
                    print(f"Skipping file {filename} (invalid JSON): {e}")
                    continue

                # Extract 'min_price' from the 'output' key
                actual_min_price = data.get("output", {}).get("min_price")
                print(f"Extracted min_price from {filename}: {actual_min_price}")  # Debug

                # Check if the extracted min_price does NOT match any y value in the array
                if actual_min_price not in y_values_set:
                    os.remove(filepath)
                    deleted_count += 1
                    print(f"Deleted file: {filename} (min_price: {actual_min_price})")
                else:
                    print(f"File {filename} matches a y value in the min_price_array.")
        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")

    # Count remaining files in the directory
    remaining_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    print(f"Number of files remaining in the directory: {len(remaining_files)}")
    print(f"Number of files deleted: {deleted_count}")

# Usage
directory_path = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Datasets/DatasetsImprovedGini/DsWordpress_20_7_improved_Gini"  # Replace with your directory path
# Replace with your array of tuples
min_price_array = [(579, 3121), (569, 2448), (415, 4281), (414, 4248), (368, 3170), (296, 4596), (276, 6586), (269, 4456), (257, 2379), (252, 4287), (247, 7728), (187, 2937), (179, 4850), (175, 6651), (169, 4503), (149, 6622), (143, 4609), (135, 4151), (123, 7806), (114, 4674), (110, 6962), (109, 4681), (101, 6674)]
delete_files_without_matching_min_price(directory_path, min_price_array)
