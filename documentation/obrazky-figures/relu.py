import torch
import matplotlib.pyplot as plt

# Define the ReLU function using PyTorch
def relu(x):
    return torch.relu(x)
# Generate input data: a range of values from -10 to 10
x = torch.linspace(-1, 1, 100)
y = relu(x)

# Plot the ReLU function
plt.figure(figsize=(8, 6))
plt.plot(x.numpy(), y.numpy(), label='ReLU Function', linewidth=5)
plt.xlabel('Input', fontsize=15)
plt.ylabel('Output', fontsize=15)
plt.title('ReLU Function', fontsize=18)
plt.legend(fontsize=13)
plt.grid(True)

# Save the plot as a vector PDF
plt.savefig('relu.pdf', format='pdf')

# Show plot in this notebook (for visualization here)
plt.show()

