import numpy as np

# Load the data from the text file
with open('outputAoA.txt', 'r') as f:
    data = f.read().split(',')
    data = [float(d) for d in data]

# Calculate the standard deviation
std = np.std(data)

# Calculate the squared standard deviation (variance)
variance = std**2

print('Standard deviation:', std)
print("Squared standard deviation :", variance)

