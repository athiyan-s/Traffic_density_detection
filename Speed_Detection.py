import cv2
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=5):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        else:
            input_centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in rects]
            if not self.objects:
                for centroid in input_centroids:
                    self.register(centroid)
            else:
                object_ids = list(self.objects.keys())
                object_centroids = list(self.objects.values())
                distances = np.linalg.norm(np.array(object_centroids)[:, None] - np.array(input_centroids)[None, :], axis=2)
                rows = distances.min(axis=1).argsort()
                cols = distances.argmin(axis=1)[rows]
                used_rows = set()
                used_cols = set()

                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue

                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)

                unused_rows = set(range(distances.shape[0])).difference(used_rows)
                unused_cols = set(range(distances.shape[1])).difference(used_cols)

                if distances.shape[0] >= distances.shape[1]:
                    for row in unused_rows:
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
                else:
                    for col in unused_cols:
                        self.register(input_centroids[col])

# Initialize the tracker
tracker = CentroidTracker()

haar_cascade = 'src/HaarCascadeClassifier.xml'
video = 'src/clip4.mp4'

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
car_cascade = cv2.CascadeClassifier(haar_cascade)

# Set the desired width and height for the video window
window_width, window_height = 800, 600

# Initialize variables for calculating stable average
window_size = 10
vehicle_counts = []

while True:
    ret, frames = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Convert 'cars' to a list of detections to avoid the truth value ambiguity error
    cars_list = np.array(cars).tolist()

    # Update the tracker with new detections
    tracker.update(cars_list)

    # Draw bounding boxes and IDs for tracked vehicles
    for object_id, centroid in tracker.objects.items():
        x, y = centroid
        cv2.putText(frames, f'ID: {object_id}', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frames, (x, y), 4, (0, 255, 0), -1)

    # Resize the frames to the desired width and height before displaying
    resized_frames = cv2.resize(frames, (window_width, window_height))

    cv2.imshow('video', resized_frames)

    # Update the vehicle count and store it in the list for calculating stable average
    vehicle_count = len(tracker.objects)
    vehicle_counts.append(vehicle_count)

    # Limit the list size to the window size for the stable average calculation
    if len(vehicle_counts) > window_size:
        vehicle_counts = vehicle_counts[-window_size:]

    # Calculate the stable average
    avg_vehicle_count = sum(vehicle_counts) / len(vehicle_counts)
    print(f"Total Vehicles: {avg_vehicle_count:.2f}")

    # Set the waitKey delay to play the video at its actual frame rate
    # The delay is in milliseconds (1000ms = 1s), so we divide by the frame rate to get the correct delay
    delay = int(1000 / fps)
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()
