import os
import csv
import re

# Function to extract parameters from filenames
def extract_parameters(filename):
    # Adjust patterns based on your use case
    rgcn_pattern = (
        r"Wordpress4_RGCN_(?P<samples>\d+)_samples_(?P<epochs>\d+)_epochs_(?P<batchsize>\d+)_batchsize_off_(?P<offers>\d+)_(?P<type>lex|nones)"
    )
    container_pattern = r"Wordpress4_off_(?P<offers>\d+)\.out"

    rgcn_match = re.match(rgcn_pattern, filename)
    if rgcn_match:
        return rgcn_match.groupdict(), "rgcn"

    container_match = re.match(container_pattern, filename)
    if container_match:
        return {"offers": container_match.group("offers")}, "container"

    return None, None

import re

def extract_metrics(filepath):
    time, price = None, None
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Look for total time
    for line in lines:
        if ":total-time" in line:
            match = re.search(r":total-time\s+([\d.]+)", line)
            if match:
                time = float(match.group(1))

    # Look for the price in the objectives section
    objectives_section = False
    for line in lines:
        # Start parsing after finding "(objectives"
        if "(objectives" in line:
            objectives_section = True

        if objectives_section:
            match = re.search(r"\)\s+(\d+)", line)
            if match:
                price = int(match.group(1))  # Convert to int since price is an integer
                break  # Stop searching after finding the price

    return time, price

# Function to process the directory and extract all data
def process_directory(directory):
    data = []
    container_data = {20: [], 27: [], 40: [], 250: [], 500: []}

    for filename in os.listdir(directory):
        # Process only files with .out extension
        if not filename.endswith(".out"):
            continue

        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue

        # Extract parameters from filename
        params, file_type = extract_parameters(filename)
        if not params:
            continue

        # Extract time and price from the file
        time, price = extract_metrics(filepath)

        # Prepare data entry
        entry = {
            "samples": int(params["samples"]) if params.get("samples") else "N/A",
            "epochs": int(params["epochs"]) if params.get("epochs") else "N/A",
            "batchsize": int(params["batchsize"]) if params.get("batchsize") else "N/A",
            "offers": int(params["offers"]),
            "type": params.get("type", "N/A"),
            "time": time if time is not None else "N/A",
            "price": price if price is not None else "N/A"
        }

        if file_type == "rgcn":
            data.append(entry)
        elif file_type == "container":
            container_data[entry["offers"]].append(entry)

    # Sort the data by offers
    data.sort(key=lambda x: x["offers"])

    # Add N/A entries before the matching container data
    final_data = []
    for offers in sorted(container_data.keys()):
        for entry in container_data[offers]:
            final_data.append({
                "samples": "N/A", "epochs": "N/A", "batchsize": "N/A", "offers": offers,
                "type": "N/A", "time": entry["time"], "price": entry["price"]
            })
        final_data.extend([item for item in data if item["offers"] == offers])

    return final_data

# Function to write extracted data to a CSV file
def write_data_to_csv(data, output_file):
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Samples", "Epochs", "Batchsize", "Offers", "Type", "Time", "Price"])

        # Write data rows
        for entry in data:
            writer.writerow([
                entry["samples"], entry["epochs"], entry["batchsize"], entry["offers"], entry["type"],
                entry["time"], entry["price"]
            ])

# Main function
def main():
    # Hardcoded input directory and output file
    input_directory = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Output/SMT-LIB/Wordpress4/"  # Replace with your directory path
    output_file = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/utils/output.csv"  # Replace with your desired output file path

    # Process directory and extract data
    data = process_directory(input_directory)

    # Write data to CSV
    write_data_to_csv(data, output_file)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    main()