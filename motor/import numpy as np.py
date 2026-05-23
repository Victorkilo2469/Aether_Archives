import numpy as np
import cv2

# Video source (0 for webcam or "video.mp4")
cap = cv2.VideoCapture(1)

# PSO parameters
num_particles = 30
iterations = 20   # lower for real-time
w = 0.7
c1 = 1.5
c2 = 1.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Initialize particles
    positions = np.random.rand(num_particles, 2)
    positions[:, 0] *= height
    positions[:, 1] *= width

    velocities = np.random.randn(num_particles, 2)

    pbest_positions = positions.copy()
    pbest_values = np.array([gray[int(p[0]), int(p[1])] for p in positions])

    gbest_index = np.argmax(pbest_values)
    gbest_position = pbest_positions[gbest_index].copy()

    # Run PSO iterations for this frame
    for _ in range(iterations):
        for i in range(num_particles):

            r1, r2 = np.random.rand(), np.random.rand()

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )

            positions[i] += velocities[i]

            # Clamp to image bounds
            positions[i][0] = np.clip(positions[i][0], 0, height - 1)
            positions[i][1] = np.clip(positions[i][1], 0, width - 1)

            value = gray[int(positions[i][0]), int(positions[i][1])]

            if value > pbest_values[i]:
                pbest_values[i] = value
                pbest_positions[i] = positions[i].copy()

        gbest_index = np.argmax(pbest_values)
        gbest_position = pbest_positions[gbest_index].copy()

    # Draw particles
    for p in positions:
        cv2.circle(frame, (int(p[1]), int(p[0])), 2, (255, 0, 0), -1)

    # Draw global best
    cv2.circle(frame,
               (int(gbest_position[1]), int(gbest_position[0])),
               6, (0, 0, 255), 2)

    cv2.imshow("PSO Brightest Point Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()