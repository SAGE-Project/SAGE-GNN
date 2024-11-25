import numpy as np


def compute_gini(values):
    """
    Compute the Gini coefficient for a list of values.
    """
    values = np.array(values)
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
#[(nr aparitii a, valorii), ...]
data = [(1156, 1777), (1038, 3216), (819, 1846), (598, 3078), (579, 3121), (569, 2448), (415, 4281), (414, 4248), (368, 3170), (296, 4596), (276, 6586), (269, 4456), (257, 2379), (252, 4287), (247, 7728), (187, 2937), (179, 4850), (175, 6651), (169, 4503), (149, 6622), (143, 4609), (135, 4151), (123, 7806), (114, 4674), (110, 6962), (109, 4681), (101, 6674), (98, 4177), (97, 4876), (78, 4223), (66, 4738), (61, 5020), (59, 7350), (58, 4794), (56, 7011), (52, 8238), (45, 4321), (41, 7190), (34, 9877), (34, 7296), (30, 8580), (30, 4962), (29, 4778), (28, 4435), (24, 5134), (16, 12699), (16, 10484), (15, 9785), (12, 17885), (11, 10435), (7, 20671), (7, 10668), (6, 19415), (6, 18535), (5, 14112), (4, 22880), (3, 20996), (3, 12316), (3, 12083), (2, 25849), (2, 22578), (2, 22358), (2, 20794), (2, 19820), (2, 16584), (1, 27000), (1, 21820), (1, 20183), (1, 19886), (1, 19587)]
# Reduce Gini coefficient
remaining_data, removed_data, final_gini = reduce_gini(data, target_gini=0.3)

# Print results
print("Remaining Data:", len(remaining_data), remaining_data)
print("Removed Data:", len(removed_data), removed_data)
print("Final Gini Coefficient:", final_gini)
