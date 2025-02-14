import numpy as np  # Import NumPy for mathematical operations and array manipulations
import matplotlib.pyplot as plt  # Import Matplotlib for visualization
import matplotlib.animation as animation  # Import Matplotlib's animation module for dynamic visualizations

# Define the quadratic function f(x) = x^2 + 2x + 1
def f(x):
    return x**2 + 2*x + 1  # This represents a parabola that opens upwards

# Define the derivative of the function f'(x) = 2x + 2
def df(x):
    return 2*x + 2  # This is used to compute the gradient at any given point

# Implement the gradient descent algorithm
def gradient_descent(learning_rate, iterations, start_point):
    x = start_point  # Initialize the starting point for the optimization
    history = []  # Create a list to store the trajectory of (x, f(x))
    for i in range(iterations):  # Iterate for the given number of steps
        history.append((x, f(x)))  # Append the current point and function value to history
        gradient = df(x)  # Compute the gradient at the current point
        x -= learning_rate * gradient  # Update x by moving in the opposite direction of the gradient
    return history  # Return the trajectory for visualization

# Function to visualize the optimization trajectory
def visualize(history, learning_rate, start_point):
    x_val = np.linspace(-5, 3, 100)  # Generate 100 points between -5 and 3 for plotting
    y_val = f(x_val)  # Compute the function values for the generated points

    # Extract x and y values from the optimization history
    history_x = [h[0] for h in history]  # Extract x values from history
    history_y = [h[1] for h in history]  # Extract f(x) values from history

    plt.figure(figsize=(10, 5))  # Create a figure with a specified size

    # Plot the function curve
    plt.plot(x_val, y_val, label="f(x) = x^2 + 2x + 1", color="blue")

    # Plot the optimization trajectory as red points and a dashed line
    plt.scatter(history_x, history_y, color="red", label="Optimization trajectory",marker="x")
    plt.plot(history_x, history_y, color="red", linestyle="--", alpha=0.6)

    # Highlight the global minimum analytically found at x = -1
    global_min = -1  # Global minimum of f(x)
    plt.scatter(global_min, f(global_min), color="green", s=100, label="Global minimum (x=-1)",marker="x")

    # Add labels, legend, and title to the plot
    plt.xlabel("x")  # Label for the x-axis
    plt.ylabel("f(x)")  # Label for the y-axis
    plt.title(f"Gradient Descent with Learning Rate={learning_rate}, Start Point={start_point}, iteration={len(history)}")
    plt.legend()  # Add a legend for clarity
    plt.grid()  # Add a grid for better readability
    plt.show()  # Display the plot

# Function to create an animation of the optimization process
def create_animation(history):
    x_val = np.linspace(-5, 3, 100)  # Generate x values for plotting the function
    y_val = f(x_val)  # Compute the function values

    fig, ax = plt.subplots(figsize=(10, 5))  # Create a subplot for the animation
    ax.plot(x_val, y_val, label="f(x) = x^2 + 2x + 1", color="blue")  # Plot the function

    # Highlight the global minimum analytically determined
    global_min = -1  # Global minimum of f(x)
    ax.scatter(global_min, f(global_min), color="green", s=100, label="Global minimum (x=-1)")

    point, = ax.plot([], [], "ro", label="Optimization trajectory")  # Create a point to represent the trajectory

    # Initialize the animation by clearing the point
    def init():
        point.set_data([], [])
        return point,

    # Update the animation frame-by-frame
    def update(frame):
        x, y = history[frame]  # Get the x and y values from the current frame of history
        point.set_data(x, y)  # Update the point position
        return point,

    # Create the animation using the update function
    ani = animation.FuncAnimation(fig, update, frames=len(history), init_func=init, blit=False, repeat=False)

    # Set labels and titles for the plot
    ax.set_xlabel("x")  # Label for the x-axis
    ax.set_ylabel("f(x)")  # Label for the y-axis
    ax.set_title("Gradient Descent Optimization Trajectory")  # Title for the animation
    ax.legend()  # Add a legend
    ax.grid()  # Add a grid for better readability
    plt.show()  # Display the animation
    return ani  # Return the animation object




# we trying to  different parameters for gradient descent
learning_rates = [0.01, 0.1, 0.5]  # Define a list of learning rates to experiment with
starting_points = [-3, 0, 3]  # Define a list of starting points
iterations_list = [10, 50, 100]  # Define a list of iteration counts

# Loop each combination of parameters
for learnR in learning_rates:
    for sp in starting_points:
        for iter_count in iterations_list:
            print(f"Learning Rate={learnR}, Start Point={sp}, Iterations={iter_count}")
            history = gradient_descent(learnR, iter_count, sp)  # Perform gradient descent

            # Visualize the trajectory
            visualize(history, learnR, sp)

            # Create an animation of the trajectory
           # ani = create_animation(history)

# Compute and display the analytical minimum
analytical_min = -1  # Analytical solution: f'(x) = 2x + 2 => x = -1
print(f"Analytical Minimum: x = {analytical_min}, f(x) = {f(analytical_min)}")  # Display the analytical result
