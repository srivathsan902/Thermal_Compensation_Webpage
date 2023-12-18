import sys

if len(sys.argv) >= 6:
    # Get the input type, value, selected option, and slider values from command-line arguments
    input_type = sys.argv[1]
    input_value = (sys.argv[2])
    selected_option = sys.argv[3]
    slider1_value = int(sys.argv[4])
    slider2_value = int(sys.argv[5])

    # Print the received values
    print(f" Selected Option: {selected_option}, Slider 1 Value: {slider1_value}, Slider 2 Value: {slider2_value}")
else:
    print("Insufficient command-line arguments.")
