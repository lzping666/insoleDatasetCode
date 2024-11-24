import numpy as np
import matplotlib.pyplot as plt

def visualize_foot_pressure(npz_file_path):
    # Load the .npz file
    data = np.load(npz_file_path)

    # Extract foot pressure data
    foot_pressure = data['foot_pressure']

    # Create a heatmap of the foot pressure
    plt.figure(figsize=(10, 6))
    plt.imshow(foot_pressure.mean(axis=0), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Foot Pressure')

    # Label the axes
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    # Add a title
    plt.title('Average Foot Pressure Heatmap')

    # Display the plot
    plt.show()

def visualize_pressure_surface(npz_file_path):
    data = np.load(npz_file_path)
    foot_pressure = data['foot_pressure']
    frame_index = 0
    pressure_data = foot_pressure[frame_index]

    x = np.arange(pressure_data.shape[1])
    y = np.arange(pressure_data.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, pressure_data, cmap='viridis')

    ax.set_title(f'Foot Pressure Surface - Frame {frame_index}')
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Pressure (Pa)')
    plt.show()





# Example usage
# npz_file_path = './output/jumpingjacks_foot_pressure.npz'

# visualize_pressure_surface(npz_file_path)
# visualize_foot_pressure(npz_file_path)