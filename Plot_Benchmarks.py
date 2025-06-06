import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# OBTAIN BENCHMARK DATA
csv_file_path = './Benchmark_Plotting/MATD3_Benchmarks.csv'  # REPLACE WITH BENCHMARK
df = pd.read_csv(csv_file_path)

# PLOT DATA
def plot_data(df, *columns, vertical_line_x, vertical_line_label):

    plt.figure(figsize=(10, 6))

    # Define colormap for groups
    def_colors = [cm.Greens(i) for i in np.linspace(0.4, 1, 4)]  
    int_colors = [cm.Reds(i) for i in np.linspace(0.4, 1, 4)] 

    for column in columns:
        if column in df.columns:
            if "Defender" in column:
                color = def_colors.pop(0)
            elif "Intruder" in column:
                color = int_colors.pop(0)
            plt.plot(df["Episode"], df[column], label=column, color=color)
            # plt.plot(df["Episode"], df[column], label=column)
        else:
            print(f"Warning: Column '{column}' does not exist in the DataFrame.")
    
    if vertical_line_x is not None:
        plt.axvline(x=vertical_line_x, color='blue', linestyle='--')
        plt.text(vertical_line_x, -14, vertical_line_label, color='blue', fontsize=10, 
         ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    # Customize the plot
    plt.xlabel('Episode', fontsize=18)
    plt.ylabel('100 Episode Averaged Fitnesses', fontsize=18)  # CHANGE FOR TYPE OF VALUE
    plt.title('MATD3 Agent Model Convergence', fontsize=22) # CHANGE FOR TYPE OF VALUE
    plt.xlim(left=20, right=2250)
    plt.ylim(bottom=-15)
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(fontsize=18, loc="lower right")
    plt.grid()
    
    # SAVE PLOT TO DIRECTORY
    output_filename = 'Convergence_Plot.png'  # NAME PLOT OUTPUT FILE
    plt.savefig(output_filename, format='png')  
    plt.close()

# EXTRACT COLUMN
columns_to_plot = ["Model 1", "Model 2", "Model 3", "Model 4"]  
plot_data(df, *columns_to_plot, vertical_line_x=None, vertical_line_label="Convergence Threshold")  # Set vertical to convergence point