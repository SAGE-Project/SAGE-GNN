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
    time_mismatch_count = {20: 0, 27: 0, 40: 0, 250: 0, 500: 0}
    price_mismatch_count = {20: 0, 27: 0, 40: 0, 250: 0, 500: 0}

    # Initialize lists to store mismatches
    time_mismatches = {20: [], 27: [], 40: [], 250: [], 500: []}
    price_mismatches = {20: [], 27: [], 40: [], 250: [], 500: []}

    # Reference values for comparison
    reference_values = {
        20: {'time': 0.3, 'price': 3759.0},
        27: {'time': 0.25, 'price': 2400.0},
        40: {'time': 0.62, 'price': 2676.0},
        250: {'time': 3.15, 'price': 1622.0},
        500: {'time': 10.38, 'price': 1582.0}
    }

    # Create lists for rows by offer group
    offer_groups = {20: [], 27: [], 40: [], 250: [], 500: []}

    # Group rows by offer
    for row in rows:
        if not is_na(row["Price"]) and not is_na(row["Time"]):  # Only process valid rows
            offer = int(row["Offers"])
            if offer in offer_groups:
                offer_groups[offer].append(row)

    # Open the output file in write mode
    with open(output_file, mode='w') as outfile:
        # Redirect print statements to the output file
        def print_to_file(*args, **kwargs):
            print(*args, file=outfile, **kwargs)

        # Iterate through offer groups and check time and price conditions
        for offer, rows_in_group in offer_groups.items():
            reference_time = reference_values[offer]['time']
            reference_price = reference_values[offer]['price']

            # Compare each row within the same offer group
            for i, row in enumerate(rows_in_group):
                current_time = float(row["Time"])
                current_price = float(row["Price"])

                # Compare price to reference price
                if current_price != reference_price:
                    price_mismatch_count[offer] += 1
                    price_mismatches[offer].append(f"Price mismatch at line {i + 1}: {row}")

                # Compare time to reference time
                if current_time > reference_time:
                    time_mismatch_count[offer] += 1
                    time_mismatches[offer].append(f"Time mismatch at line {i + 1}: {row}")

        # Print results with reference values
        for offer in [20, 27, 40, 250, 500]:
            reference_time = reference_values[offer]['time']
            reference_price = reference_values[offer]['price']

            # Print price mismatch details
            print_to_file(
                f"\nPrice ({reference_price}) mismatch count for offers {offer}: {price_mismatch_count[offer]} out of {len(offer_groups[offer])} comparisons")
            for mismatch in price_mismatches[offer]:
                print_to_file(mismatch)

            # Print time mismatch details
            print_to_file(
                f"\nTime ({reference_time}) mismatch count for offers {offer}: {time_mismatch_count[offer]} out of {len(offer_groups[offer])} comparisons")
            for mismatch in time_mismatches[offer]:
                print_to_file(mismatch)


# Main function
def main():
    input_file = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/utils/generate_statistics_in_csv_secure_web.csv"  # Replace with the path to your CSV file
    output_file = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/utils/generate_statistics_time_price_from_csv.out"  # Replace with the path to your CSV file
    process_csv(input_file, output_file)


if __name__ == "__main__":
    main()
