import numpy as np


def compute_gini(values):
    """
    Compute the Gini coefficient for a list of values.
    """
    values = np.array(values)
    # needs sorting in ascending orderc
    values_sorted = np.sort(values)
    n = len(values)
    cumulative_values = np.cumsum(values_sorted)
    gini = (1 / n) * (n + 1 - 2 * np.sum(cumulative_values) / cumulative_values[-1])
    return gini


def reduce_gini(data, target_gini=0.3):
    """
    Reduce the Gini coefficient to the target value by removing tuples.

    Parameters:
    - data: List of tuples [(count, value), ...]
    - target_gini: Target Gini coefficient (default is 0.3)

    Returns:
    - remaining_data: List of tuples that remain after reduction
    - removed_data: List of tuples that were removed
    - final_gini: Final Gini coefficient after reduction
    """
    # Sort data by count in descending order
    data_sorted = sorted(data, key=lambda x: x[0], reverse=True)
    remaining_data = data_sorted[:]
    removed_data = []

    # Extract only the counts
    counts = [count for count, value in remaining_data]

    # Compute initial Gini coefficient
    current_gini = compute_gini(counts)

    while current_gini > target_gini and remaining_data:
        # Remove the tuple with the largest count
        removed_tuple = remaining_data.pop(0)
        removed_data.append(removed_tuple)

        # Recompute the Gini coefficient with the updated list
        counts = [count for count, value in remaining_data]
        current_gini = compute_gini(counts) if counts else 0  # Avoid division by zero

    return remaining_data, removed_data, current_gini


# Example data
data = [(966, 3808), (946, 5531), (911, 5057), (649, 6865), (647, 5083), (628, 3952),
        (604, 5583), (506, 3759), (427, 5227), (422, 5871), (409, 4066), (407, 3785),
        (369, 4609), (364, 7150), (284, 5341), (284, 3929), (282, 4635), (264, 4146),
        (263, 6099), (200, 7350), (194, 4779), (194, 4043), (182, 5421), (180, 6259) ]
print(len(data))
# Reduce Gini coefficient
remaining_data, removed_data, final_gini = reduce_gini(data, target_gini=0.3)

# Write results to a file
with open("../Output/Stats_ImproveGini/???.txt", "w") as file:
    file.write(f"Remaining Data ({len(remaining_data)} items):\n{remaining_data}\n\n")
    file.write(f"Removed Data ({len(removed_data)} items):\n{removed_data}\n\n")
    file.write(f"Final Gini Coefficient: {final_gini}\n")