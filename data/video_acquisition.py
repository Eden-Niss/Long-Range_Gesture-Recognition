import cv2
import time
import re
import os


parent_dir = r'/home/roblab20/PycharmProjects/LongRange/data/videos/'
gest_class = r'Bad'  # 'Bad', 'Come', 'Good', 'None', 'Point', 'Stop',
distance = 1
video_path = parent_dir + gest_class + '/' + str(distance)

# Define the number of iterations
num_iterations = 25

# Define the duration of each video capture in seconds
duration = 4

to_save = True
# to_save = False


# =================================================================


def video_name(video_path, gest_class, i):
    img_name = video_path + '/' + gest_class + '_{}'.format(i)

    tt = str(time.asctime())
    img_name_save = (img_name + " " + str(re.sub('[:!@#$]', '_', tt) + '.avi')).replace(' ', '_')

    return img_name_save


os.makedirs(video_path, exist_ok=True)

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change to a different number if you have multiple cameras

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Loop for the specified number of iterations
for i in range(num_iterations):
    # Create a unique output file name for each iteration
    output_filename = video_name(video_path, gest_class, i)
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

    # Record video for the specified duration
    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Write the frame to the output video
        if to_save:
            out.write(frame)

        # Display the frame (optional)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoWriter object for this iteration
    out.release()

# Release the VideoCapture object when done
cap.release()
cv2.destroyAllWindows()
