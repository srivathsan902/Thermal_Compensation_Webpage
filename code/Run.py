import os
import pandas as pandas
from data_processing import output
import sys

if len(sys.argv) >= 6:
    # Get the input type, value, selected option, and slider values from command-line arguments
    input_type = sys.argv[1]
    input_value = (sys.argv[2])
    model = sys.argv[3]
    error_threshold = int(sys.argv[4])
    test_split = int(sys.argv[5])

    # Print the received values
    # print(f" Selected Option: {model}, Slider 1 Value: {error_threshold}, Slider 2 Value: {test_split}")
else:
    print("Insufficient command-line arguments.")

script_directory = os.path.dirname(os.path.abspath(__file__))

folder_path = os.path.join(script_directory, 'Datasets')
file_paths=[]

for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Construct the full path to the file
            file_path = os.path.join(root, file)
            # Append the file path to the list
            file_paths.append(file_path)

file1=file_paths[0]
file2=file_paths[1]

fn=output(model,file1,file2,error_threshold,test_split)
print(fn)
