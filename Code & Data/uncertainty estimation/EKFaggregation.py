import numpy as np

# Read data from input files
filenames = ['EKF1_results.txt', 'EKF2_results.txt', 'EKF3_results.txt', 'EKF4_results.txt', 'EKF5_results.txt']
num_files = len(filenames)
data = []
for filename in filenames:
    with open(filename, 'r') as f:
        file_data = np.loadtxt(f, delimiter=',')
        data.append(file_data)

# Calculate the sum of values for each row across all input files
sum_data = np.sum(data, axis=0)

# Calculate the average of values for each row
avg_data = sum_data / num_files

# Write the aggregated results to output file
with open('EKFagg_results.txt', 'w') as f:
    for row in avg_data:
        row_str = ', '.join(map(lambda x: '{:.10f}'.format(x), row))
        f.write(row_str + '\n')
