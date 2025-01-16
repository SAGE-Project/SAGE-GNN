import csv

# Function to check if a value is 'N/A'
def is_na(value):
    return value == 'N/A'


# Function to read the CSV and process comparisons
def process_csv(input_file, output_file):
    with open(input_file, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Initialize counters for mismatches
    time_mismatch_count = {20: 0, 40: 0, 250: 0, 500: 0}
    price_mismatch_count = {20: 0, 40: 0, 250: 0, 500: 0}

    # Initialize counters for smaller than reference values
    price_smaller_count = {20: 0, 40: 0, 250: 0, 500: 0}
    time_smaller_count = {20: 0, 40: 0, 250: 0, 500: 0}

    # Initialize lists to store mismatches
    time_mismatches = {20: [], 40: [], 250: [], 500: []}
    price_mismatches = {20: [], 40: [], 250: [], 500: []}

    # Reference values for comparison
    reference_values = {
        20: {'time': 1.45, 'price': 1777},
        27: {'time': 0.46, 'price': 1440},
        40: {'time': 5.05, 'price': 1396},
        250: {'time': 28.62, 'price': 1260},
        500: {'time': 146.13, 'price': 1210}
    }

    # Create lists for rows by offer group
    offer_groups = {20: [], 40: [], 250: [], 500: []}

    # Group rows by offer
    for row in rows:
        if not is_na(row["Price"]) and not is_na(row["Time"]):  # Only process valid rows
            try:
                offer = int(row["Offers"])
                if offer in offer_groups:
                    offer_groups[offer].append(row)
            except ValueError:
                print(f"Skipping invalid offer: {row['Offers']}")

    # Open the output file in write mode
    with open(output_file, mode='w') as outfile:
        # Redirect print statements to the output file
        def print_to_file(*args, **kwargs):
            print(*args, file=outfile, **kwargs)

        # Iterate through offer groups and check time and price conditions
        for offer, rows_in_group in offer_groups.items():
            if offer not in reference_values:
                print_to_file(f"Warning: No reference values for offer {offer}")
                continue  # Skip processing if no reference value for this offer

            reference_time = reference_values[offer]['time']
            reference_price = reference_values[offer]['price']

            # Initialize counters for current group
            group_time_mismatch_count = 0
            group_price_mismatch_count = 0
            group_time_smaller_count = 0
            group_price_smaller_count = 0

            # Process each row for this offer group
            for i, row in enumerate(rows_in_group):
                current_time = float(row["Time"])
                current_price = float(row["Price"])

                # Compare price to reference price
                if current_price != reference_price:
                    group_price_mismatch_count += 1
                    price_mismatches[offer].append(f"Price mismatch at line {i + 1}: {row}")

                # Compare time to reference time
                if current_time > reference_time:
                    group_time_mismatch_count += 1
                    time_mismatches[offer].append(f"Time mismatch at line {i + 1}: {row}")

                # Check if current price is smaller than the reference price for this offer
                if current_price < reference_price:
                    group_price_smaller_count += 1

                # Check if current time is smaller than the reference time for this offer
                if current_time < reference_time:
                    group_time_smaller_count += 1

            # Print only once for each offer group
            print_to_file(
                f"Price ({reference_price}) mismatch count for offers {offer}: {group_price_mismatch_count} out of {len(rows_in_group)} comparisons")
            for mismatch in price_mismatches[offer]:
                print_to_file(mismatch)

            print_to_file(
                f"Time ({reference_time}) mismatch count for offers {offer}: {group_time_mismatch_count} out of {len(rows_in_group)} comparisons")
            for mismatch in time_mismatches[offer]:
                print_to_file(mismatch)

# Main function
def main():
    input_file = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/utils/generate_statistics_in_csv_wordpress3.csv"  # Replace with the path to your CSV file
    output_file = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/utils/generate_statistics_time_price_from_csv_wordpress3.out"  # Replace with the path to your CSV file
    process_csv(input_file, output_file)


if __name__ == "__main__":
    main()
