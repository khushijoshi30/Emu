# Emu
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Load the P&ID diagram
image_path = 'pid_diagram.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Preprocessing the Image
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection to detect edges
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Step 2: Detecting Lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Create a copy of the image to draw the lines
line_image = np.copy(image)

# Initialize the graph to store lines and intersections
G = nx.Graph()

# Function to compute direction between two points
def get_direction(x1, y1, x2, y2):
    if x1 == x2:
        return 'Vertical' if y2 > y1 else 'Upwards'
    elif y1 == y2:
        return 'Horizontal' if x2 > x1 else 'Leftwards'
    else:
        return 'Diagonal'

# Step 3: Draw the detected lines and add them to the graph
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw lines in blue
        
        # Add the line as an edge in the graph
        G.add_edge((x1, y1), (x2, y2), direction=get_direction(x1, y1, x2, y2))

# Step 4: Path Extraction and Handling Multiple Paths
# Finding all paths in the graph
paths = list(nx.all_simple_paths(G, source=(x1, y1), target=(x2, y2)))

# Step 5: Visualizing the Results
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Diagram with Detected Lines')
plt.imshow(line_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Detected Paths')
plt.imshow(image, cmap='gray')

# Visualize paths in different colors
for i, path in enumerate(paths):
    color = tuple(np.random.randint(0, 255, size=3).tolist())  # Random color for each path
    for node in path:
        cv2.circle(image, node, 5, color, -1)
        
plt.show()

# Step 6: Extract Directions for Each Path
for path in paths:
    directions = []
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        directions.append(get_direction(x1, y1, x2, y2))
    print(f"Path {path} has directions: {directions}")
