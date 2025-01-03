import csv


# Function to filter and remove unwanted lines
def remove_unwanted_lines(input_file, output_file):
    # Open the input file and read all lines
    with open(input_file, mode='r') as infile:
        lines = infile.readlines()

    # Open the output file in write mode
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)

        # Iterate through the lines and write only the lines that do not start with the specified patterns
        for line in lines:
            if not (line.startswith("(assert-soft (= PriceProv") or
                    line.startswith("(assert-soft (= MemProv") or
                    line.startswith("(assert-soft (= ProcProv") or
                    line.startswith("(assert-soft (= StorageProv")):
                outfile.write(line)  # Write the filtered line to the output file

                # Main function to execute the process


def main():
    input_file  = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Output/SMT-LIB/SecureWebContainer/SecureWeb_RGCN_1000_samples_1000_epochs_32_batchsize_off_27_lex"  # Replace with your actual input file path
    output_file = "/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Output/SMT-LIB/SecureWebContainer/SecureWeb_RGCN_1000_samples_1000_epochs_32_batchsize_off_27_lex-processed.out"  # Replace with your desired output file path
    remove_unwanted_lines(input_file, output_file)


if __name__ == "__main__":
    main()
