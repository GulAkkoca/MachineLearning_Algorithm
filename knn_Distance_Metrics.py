#Gül Akkoca
import numpy as np
from sklearn.datasets import load_iris  # Importing Iris dataset
from sklearn.model_selection import train_test_split  # Importing function to split dataset
from sklearn.preprocessing import MinMaxScaler  # Importing scaler to normalize features
from sklearn.metrics import accuracy_score  # Importing accuracy metric for evaluation
from collections import Counter  # Importing Counter for majority vote
import matplotlib.pyplot as plt  # Importing for plotting
from scipy.spatial import Voronoi, voronoi_plot_2d  # Importing Voronoi diagram tools

# Step 1: Load and preprocess the Iris dataset
def load_and_preprocess():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features of the dataset
    y = iris.target  # Target labels of the dataset
    
    # Normalize features using Min-Max Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into training and testing sets (30% for testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=48  # I choosed just a random state
    )
    
    # Return the training and testing data, feature names, and target labels
    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names

# Step 2: Implement distance metrics
def compute_distance(x1, x2, metric):
    # Calculate Euclidean distance between two points
    if metric == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    # Calculate Manhattan distance between two points
    elif metric == 'manhattan':
        return np.sum(np.abs(x1 - x2))
    # Calculate Chebyshev distance between two points
    elif metric == 'chebyshev':
        return np.max(np.abs(x1 - x2))
    # Calculate Cosine distance between two points
    elif metric == 'cosine':
        return 1 - (np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
    else:
        raise ValueError("Unsupported metric!")  # Error if the metric is not supported

# Step 3: Implement KNN
def knn(X_train, y_train, X_test, k, metric):
    predictions = []  # List to store predictions for each test point
    # Loop through each test point
    for test_point in X_test:
        # Compute distances from test_point to all training points
        distances = [compute_distance(test_point, train_point, metric) for train_point in X_train]
        
        # Find the k nearest neighbors by sorting distances
        k_indices = np.argsort(distances)[:k]
        k_neighbors = [y_train[i] for i in k_indices]  # Get the corresponding labels of the k nearest neighbors
        
        # Majority vote for classification
        most_common = Counter(k_neighbors).most_common(1)
        predictions.append(most_common[0][0])  # Append the most common class label
    
    return np.array(predictions)  # Return the predictions as a numpy array

# Step 4: Test and evaluate KNN
def evaluate_knn(X_train, y_train, X_test, y_test, metrics, k=3):
    results = {}  # Dictionary to store accuracy for each metric
    # Loop through each metric
    for metric in metrics:
        y_pred = knn(X_train, y_train, X_test, k, metric)  # Get predictions using KNN
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
        results[metric] = accuracy  # Store accuracy in results dictionary
        print(f"Accuracy with {metric} distance: {accuracy:.2f}")  # Print accuracy for current metric
    return results  # Return the results dictionary containing accuracies

# Step 5: Visualize decision boundaries using Voronoi diagrams
def plot_voronoi(X_train, y_train, metric, feature_names, target_names):
    # Use the first two features for simplicity in 2D plotting
    X = X_train[:, :2]
    
    # Generate Voronoi diagram for the 2D data points
    vor = Voronoi(X)
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure for plotting
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1.6, line_alpha=0.6, point_size=2)
    
    # Scatter plot of data points with colors based on their class labels
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_train, cmap='viridis', edgecolor='k')
    
    # Set the title and axis labels
    ax.set_title(f"Voronoi Diagram ({metric} Distance)", fontsize=14)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    legend = ax.legend(*scatter.legend_elements(), title="Classes")  # Create a legend for the classes
    ax.add_artist(legend)  # Add the legend to the plot
    plt.show()  # Display the plot

# Main function to execute the workflow
if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_preprocess()
    
    # Define the list of metrics to evaluate
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
    
    # Evaluate KNN with each metric and print results
    print("Evaluating KNN with different metrics...")
    results = evaluate_knn(X_train, y_train, X_test, y_test, metrics)
    
    # Visualize decision boundaries for each metric using Voronoi diagrams
    for metric in metrics:
        plot_voronoi(X_train, y_train, metric, feature_names, target_names)
