import numpy as np


def npz_to_txt(npz_file_path, txt_file_path):
    # Load the .npz file
    data = np.load(npz_file_path)

    # Open the .txt file for writing
    with open(txt_file_path, 'w') as txt_file:
        # Iterate over each array in the .npz file
        for array_name in data.files:
            txt_file.write(f"Array name: {array_name}\n")
            txt_file.write(f"{data[array_name]}\n\n")


# Example usage
npz_file_path = './output/jumpingjacks_foot_pressure.npz'
txt_file_path = './output/jumpingjacks_foot_pressure.txt'
npz_to_txt(npz_file_path, txt_file_path)