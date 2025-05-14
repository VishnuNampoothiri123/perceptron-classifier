import numpy as np

# Function to normalize weights
def normalize_weights(weights):
    max_weight = np.max(weights)
    normalized_weights = (15 / max_weight) * weights
    return normalized_weights

# Function to quantize weights to 0 or 1
def quantize_weights(normalized_weights):
    quantized_weights = (normalized_weights >= 7.5).astype(int)
    return quantized_weights

# Process weights for each module
def process_weights(weights):
    normalized_weights = normalize_weights(weights)
    final_weights = quantize_weights(normalized_weights)
    return final_weights

# Example weights for 4 modules, each with 4 4-bit weights (16 weights in total)
weights = np.random.rand(16)  # Random weights between 0 and 1

# Process the weights for each module
final_weights_list = []
for i in range(4):
    module_weights = weights[i*4:(i+1)*4]
    final_weights = process_weights(module_weights)
    final_weights_list.extend(final_weights)

# Create perceptron instances for each task with final weights
final_weights = np.array(final_weights_list)
perceptrons = [final_weights[i*4:(i+1)*4] for i in range(4)]

# Example inputs to classify
inputs = [
    np.array([0, 0, 0, 1]),  # 1 in binary
    np.array([1, 0, 0, 0]),  # 8 in binary
    np.array([0, 1, 1, 1]),  # 7 in binary
    np.array([0, 0, 1, 1]),  # 3 in binary
    np.array([1, 0, 0, 1]),  # 9 in binary
    np.array([0, 0, 1, 0])   # 2 in binary
]

# Perceptron classifier function for each task
def classify_taskA(x, weights):
    weighted_sum = np.dot(x, weights)
    return 1 if weighted_sum % 2 == 0 else 0

def classify_taskB(x, weights):
    weighted_sum = np.dot(x, weights)
    return 1 if weighted_sum % 2 != 0 else 0

def classify_taskC(x, weights):
    weighted_sum = np.dot(x, weights)
    return 1 if weighted_sum >= 8 else 0

def classify_taskD(x, weights):
    weighted_sum = np.dot(x, weights)
    return 1 if weighted_sum < 8 else 0

def classify_taskE(x, weights):
    weighted_sum = np.dot(x, weights)
    return 1 if weighted_sum == 3 else 0

# Correct classifications for each input and task
correct_classifications = {
    'TaskA': [0, 1, 0, 0, 1, 1],
    'TaskB': [1, 0, 1, 1, 0, 0],
    'TaskC': [0, 1, 0, 0, 1, 0],
    'TaskD': [1, 0, 1, 1, 0, 1],
    'TaskE': [0, 0, 0, 1, 0, 0]
}

# Classify each input using each perceptron and print the results
tasks = [classify_taskA, classify_taskB, classify_taskC, classify_taskD, classify_taskE]
task_names = ['TaskA', 'TaskB', 'TaskC', 'TaskD', 'TaskE']

for task_idx, task in enumerate(tasks):
    task_name = task_names[task_idx]
    print(f"{task_name}:")
    for i, perceptron in enumerate(perceptrons):
        print(f"  Module {i + 1} Weights:", perceptron)
        for j, x in enumerate(inputs):
            result = task(x, perceptron)
            correct = correct_classifications[task_name][j]
            is_correct = 1 if result == correct else 0
            print(f"    Input {x} -> Classification: {result} (Expected: {correct}) -> Correct: {is_correct}")
