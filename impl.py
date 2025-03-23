import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt


processed_image = cv2.imread("2D_cleaned_final.png", cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread("Figures/2D.png")


scale_factor = 2  # Change this to upscale (1 = original, 2 = 2x resolution, etc.)
original_image = cv2.resize(original_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
processed_image = cv2.resize(processed_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)


MOVES = [
    (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
    (-1, -1, 1.41), (-1, 1, 1.41), (1, -1, 1.41), (1, 1, 1.41)
]

def dijkstra(image, start, goal):

    h, w = image.shape
    INF = float('inf')

    distances = np.full((h, w), INF)
    distances[start] = 0

    pq = [(0, start)]
    parent_map = {}

    while pq:
        cost, (y, x) = heapq.heappop(pq)

        if (y, x) == goal:
            break

        for dy, dx, move_cost in MOVES:
            ny, nx = y + dy, x + dx

            if 0 <= ny < h and 0 <= nx < w and image[ny, nx] == 255:
                new_cost = cost + move_cost

                if new_cost < distances[ny, nx]:
                    distances[ny, nx] = new_cost
                    heapq.heappush(pq, (new_cost, (ny, nx)))
                    parent_map[(ny, nx)] = (y, x)

    path = []
    current = goal
    while current in parent_map:
        path.append(current)
        current = parent_map[current]
    path.append(start)
    path.reverse()

    return path


start = (732 * scale_factor, 352 * scale_factor)
goal = (358 * scale_factor, 818 * scale_factor)


shortest_path = dijkstra(processed_image, start, goal)

path_array = np.array(shortest_path, dtype=np.int32)
for i in range(1, len(path_array)):
    cv2.line(original_image, tuple(path_array[i - 1][::-1]), tuple(path_array[i][::-1]), (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)

cv2.circle(original_image, (start[1], start[0]), 10, (0, 0, 255), -1)  # Red start
cv2.circle(original_image, (goal[1], goal[0]), 10, (255, 0, 0), -1)  # Blue goal

plt.figure(figsize=(12, 12))  # Increase figure size
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.savefig('Output/2D_HighRes.png', dpi=300, bbox_inches='tight')
plt.show()
