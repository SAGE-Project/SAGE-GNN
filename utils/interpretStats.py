import os
import matplotlib.pyplot as plt
from collections import Counter


# Function to parse the ":time" metric from a single file
def parse_time_metric(file_path):
    time_value = None  # Initialize time_value to None
    found_time = False
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()  # Remove any extra spaces
                if line.startswith(":time"):
                    time_str = line.split()[-1].strip(")")  # Get the time value
                    time_value = float(time_str)  # Convert to float
                    found_time = True
                    break
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    if not found_time:
        print(f":time metric not found in {file_path}")
    return time_value


# Function to categorize files and parse their ":time" metrics
def categorize_files_and_parse(directory_path):
    sorted_times = {
        'Z3': [],  # List to store times and prefixes for Z3
        'Initvals': [],  # List to store times and prefixes for Initvals
        'Pseudobool': []  # List to store times and prefixes for Pseudobool
    }

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            # Check for the prefixes
            if "Z3" in filename and filename.split("Z3", 1)[1].strip() == "":
                prefix = filename.split("Z3")[0]  # Get prefix before "Z3"
                time_value = parse_time_metric(file_path)
                if time_value is not None:  # Only add if time_value is valid
                    sorted_times['Z3'].append((prefix, time_value, filename))
            elif "initvals" in filename and filename.split("initvals", 1)[1].strip() == "":
                prefix = filename.split("initvals")[0]  # Get prefix before "initvals"
                time_value = parse_time_metric(file_path)
                if time_value is not None:  # Only add if time_value is valid
                    sorted_times['Initvals'].append((prefix, time_value, filename))
            elif "pseudobool" in filename and filename.split("pseudobool", 1)[1].strip() == "":
                prefix = filename.split("pseudobool")[0]  # Get prefix before "pseudobool"
                time_value = parse_time_metric(file_path)
                if time_value is not None:  # Only add if time_value is valid
                    sorted_times['Pseudobool'].append((prefix, time_value, filename))

    # Sort the times for each category by prefix
    for key in sorted_times:
        sorted_times[key].sort(key=lambda x: x[0])  # Sort by prefix

    return sorted_times


# Function to count how many times each category has the minimum time at each index
def count_minimum_times_at_indices(sorted_times):
    count_z3 = 0
    count_initvals = 0
    count_pseudobool = 0
    initvals_files_with_min = []  # List to store filenames for initvals with minimum time
    z3_files_with_min = []  # List to store filenames for Z3 with minimum time
    pseudobool_files_with_min = []  # List to store filenames for Pseudobool with minimum time

    max_len = max(len(sorted_times['Z3']), len(sorted_times['Initvals']), len(sorted_times['Pseudobool']))

    # Prepare lists for consistent comparison
    z3_times = sorted_times['Z3']
    initvals_times = sorted_times['Initvals']
    pseudobool_times = sorted_times['Pseudobool']

    for i in range(max_len):
        times = []
        if i < len(z3_times):
            times.append((z3_times[i][1], 'Z3', z3_times[i][2]))  # Append time, label, and filename
        if i < len(initvals_times):
            times.append((initvals_times[i][1], 'Initvals', initvals_times[i][2]))  # Append time, label, and filename
        if i < len(pseudobool_times):
            times.append(
                (pseudobool_times[i][1], 'Pseudobool', pseudobool_times[i][2]))  # Append time, label, and filename

        if times:
            min_time, min_label, min_filename = min(times,
                                                    key=lambda x: x[0])  # Find the minimum time, label, and filename
            if min_label == 'Z3':
                count_z3 += 1
                z3_files_with_min.append(min_filename)  # Track filename for Z3 minimum
            elif min_label == 'Initvals':
                count_initvals += 1
                initvals_files_with_min.append(min_filename)  # Track filename for initvals minimum
            elif min_label == 'Pseudobool':
                count_pseudobool += 1
                pseudobool_files_with_min.append(min_filename)  # Track filename for pseudobool minimum

    return count_z3, count_initvals, count_pseudobool, initvals_files_with_min, z3_files_with_min, pseudobool_files_with_min


# Function to analyze patterns in filenames for minimum times
def analyze_patterns(files):
    # Extract prefixes or other characteristics from filenames
    prefixes = [filename.split("Z3")[0] for filename in files] if "Z3" in files[0] else \
        [filename.split("initvals")[0] for filename in files] if "initvals" in files[0] else \
            [filename.split("pseudobool")[0] for filename in files]

    # Count occurrences of each prefix
    prefix_counts = Counter(prefixes)

    return prefix_counts


# Main function to run the program
def main(directory_path):
    sorted_times = categorize_files_and_parse(directory_path)

    # Print lengths of time lists
    print(f"Length of Z3 times: {len(sorted_times['Z3'])}")
    print(f"Length of Initvals times: {len(sorted_times['Initvals'])}")
    print(f"Length of Pseudobool times: {len(sorted_times['Pseudobool'])}")

    # Count minimum times at each index
    count_z3, count_initvals, count_pseudobool, initvals_files_with_min, z3_files_with_min, pseudobool_files_with_min = count_minimum_times_at_indices(
        sorted_times)

    print("\nCount of minimum times at each index:")
    print(f"Z3: {count_z3}")
    print(f"Initvals: {count_initvals}")
    print(f"Pseudobool: {count_pseudobool}")

    # List files for which each category gave the minimum
    print("\nFiles where Z3 had the minimum time:")
    for filename in z3_files_with_min:
        print(filename)

    print("\nFiles where Initvals had the minimum time:")
    for filename in initvals_files_with_min:
        print(filename)

    print("\nFiles where Pseudobool had the minimum time:")
    for filename in pseudobool_files_with_min:
        print(filename)

# Directory path containing the Z3 output data files
if __name__ == "__main__":
    directory_path = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Output/Stats/Oryx2/1HiddenLayer"  # Replace with your actual directory path
    main(directory_path)
