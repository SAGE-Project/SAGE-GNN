import os
def count_occurrences_in_files(directory, x):
    count = 0
    file_count = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):  # Adjust as needed for file types
            file_count += 1
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    content = file.read()
                    count += content.count(x)
                except (UnicodeDecodeError, IOError):
                    print(f"Error reading file: {filename}")
    return count, file_count

# Specify the path to the directory in your Google Drive
directory_path = '/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Datasets/DsSecureBillingEmail_20_7'  # Replace with your actual path
x_value = '2737'  # Replace with the string you want to search for

result, file_count = count_occurrences_in_files(directory_path, x_value)
print(f"The string '{x_value}' appears {result} times in the content of the files.")
print(f"There are {file_count} files searched in the directory.")