import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter

# SCALE PLOT
def scale_plot(x, height, N):
    if x < 0 or x > height:
        return ''  
    return round((x / height) * (2 * N) - N, 1)

# EXTRACT FEATURES

def extract_ball_centers_from_gif(gif_path, color_ranges):
    # Initialize a list to store ball centers
    ball_centers = {color_name: ([], [], [], []) for color_name in color_ranges.keys()}  # Added a size list
    start_frame = 0 # Start frame index 
    end_frame = 79 # End frame index 

    # Use imageio.get_reader to read the GIF file frame by frame
    with imageio.get_reader(gif_path) as reader:
        # Read the first frame to get the dimensions
        first_frame = reader.get_data(0)
        height, width, _ = first_frame.shape

        # Create a list to store the selected frames
        selected_frames = []

        # Iterate through the frames and select the desired section
        for i, frame in enumerate(reader):
            if start_frame <= i < end_frame:
                selected_frames.append(frame)

        # Iterate through each frame of the GIF
        for frame in selected_frames:
            # Convert to BGR for OpenCV
            img_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
            # Process each color range definition
            for color_name, (lower, upper) in color_ranges.items():
                # Create a mask for the specific color range
                mask = cv2.inRange(img_bgr, lower, upper)
                
                # Find contours of the detected colored areas
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Loop through contours to find the center and size
                for contour in contours:
                    if cv2.contourArea(contour) > 10:  # Filter small contours
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            ball_centers[color_name][0].append(cX)
                            ball_centers[color_name][1].append(height - cY)

                            # Calculate the radius based on contour area
                            radius = int(np.sqrt(cv2.contourArea(contour) / np.pi))  # Calculate the radius
                            ball_centers[color_name][2].append(radius)  # Store the radius size
                            ball_centers[color_name][3].append(frame)   # Store the frame

    return ball_centers, width, height  # Return width and height as well

def plot_ball_centers(ball_centers, color_mapping, legend_mapping, width, height, N):
    plt.figure(figsize=(width / 100, height / 100))  # Set figure size based on GIF dimensions

    # int_det_rad = 3/16 * height/4
    # def_det_rad = 5/16 * height/4
    int_det_rad = 0.001
    def_det_rad = 0.001
    
    for color_name, (x_positions, y_positions, sizes, frames) in ball_centers.items():
        if color_name == "Blue" or color_name == "Brown":
            # Plot blue and brown balls with circular outlines
            for x, y, size, frame in zip(x_positions, y_positions, sizes, frames):
                fill = color_name == "Brown"
                circle = Circle((x, y), size, color='none', ec=color_mapping[color_name], linewidth=2, fill=fill)
                plt.gca().add_patch(circle)
        else:
            # Use the defined single color for the other scatter points
            scatter_color = color_mapping[color_name]
            label = legend_mapping[color_name]
            plt.scatter(x_positions, y_positions, label=label, s=5, color=scatter_color)
            for idx, (x, y, size, frame) in enumerate(zip(x_positions, y_positions, sizes, frames)):
                if idx == 0:
                    if scatter_color == 'green':
                        circle = Circle((x, y), def_det_rad, color=scatter_color, alpha=0.5, fill=True)  # Lighter circle
                        plt.gca().add_patch(circle)
                    else:
                        circle = Circle((x, y), int_det_rad, color=scatter_color, alpha=0.5, fill=True)  # Lighter circle
                        plt.gca().add_patch(circle)


    plt.title('Agent Simulation Trajectories', fontsize=22)

    plt.xlim(0, width)  
    plt.ylim(0, height)
    ticks = np.arange(0, width, width/8)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.minorticks_on()
    formatter = FuncFormatter(lambda x, _: scale_plot(x, height, N))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Major gridlines
    plt.grid(which='minor', linestyle='-', linewidth='0.5', color='gray')  # Minor gridlines

    plt.xlabel('X Position', fontsize=18)
    plt.ylabel('Y Position', fontsize=18)
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(fontsize=18)
    
    # SAVE PLOT TO DIRECTORY
    output_filename = './Trajectory_Plotting/MATD3_BaseDef_Coop_Full_Traj.png'  # NAME PLOT OUTPUT FILE
    plt.savefig(output_filename, format='png')  
    plt.close()  

def save_points_to_csv(points_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Color', 'X', 'Y', 'Radius', 'Frame']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for point in points_list:
            writer.writerow(point)

# GIF READ PATH
gif_path = "./Videos/MATD3_Search.gif"  # ADD GIF PATH

# COLOUR DICTIONARY
color_ranges = {
    "Red": (np.array([50, 50, 100]), np.array([50, 50, 255])),    # For red balls
    "Green": (np.array([0, 100, 0]), np.array([50, 255, 50])),  # For green balls
    "Blue": (np.array([100, 0, 0]), np.array([255, 50, 50])),   # For blue balls
    "Brown": (np.array([0, 0, 100]), np.array([45, 45, 255])), # Define range for brown balls
}

color_mapping = {
    "Red": 'red',
    "Green": 'green',
    "Blue": 'blue',
    "Brown": 'black', 
}

legend_mapping = {
    "Red": 'Intruder',
    "Green": 'Defender',
    "Blue": 'Base',
    "Brown": 'Obstacle',  
}

# PLOT DATA
scale = 50000
ball_centers, width, height = extract_ball_centers_from_gif(gif_path, color_ranges)

all_points_for_csv = []

for color_name, (x_list, y_list, sizes, frames) in ball_centers.items():
    for x, y, size, frame in zip(x_list, y_list, sizes, frames):
        # Append each point as a separate entry
        all_points_for_csv.append({
            'Color': color_name,
            'X': x,
            'Y': y,
            'Radius': size
        })

csv_filename = './Trajectory_Plotting/all_points_separated.csv'
save_points_to_csv(all_points_for_csv, csv_filename)
plot_ball_centers(ball_centers, color_mapping, legend_mapping, width, height, scale)