import time
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import pygame

# Initialize pygame for sound playback
pygame.mixer.init()

# Constants and global variables
PLOT_LENGTH = 200
GLOBAL_CHEAT = 0 
PERCENTAGE_CHEAT = 0
CHEAT_THRESH = 0.6
HEAD_MOVEMENT_THRESH = 0.3
NECK_MOVEMENT_THRESH = 0.3
HAND_MOVEMENT_THRESH = 10
SCREENSHOT_INTERVAL = 5
EMPTY_CHAIR_THRESHOLD = 15  # Time in seconds to consider a chair empty for more than 1 minute
last_screenshot_time = 0

# Mobile detection constants
MOBILE_CONF_THRESHOLD = 0.5
MOBILE_USAGE_LOG_FILE = 'mobile_usage_log.txt'

# Video recording parameters
video_folder = 'recorded_videos'
os.makedirs(video_folder, exist_ok=True)
video_writer = None

# Load models
model = YOLO('yolov8x.pt')  # YOLO model for person and chair detection

# Load the YOLOv3 model for mobile detection
mobile_model = cv2.dnn.readNet("yolov3W.weights", "yolov3.cfg")

# Load class labels for YOLOv3
yolo_classes = []
with open('coco.names', 'r') as f:
    yolo_classes = f.read().splitlines()

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Dictionary to store previous positions of detected persons
previous_positions = {}
# Dictionary to store the detection time of empty chairs
empty_chair_detection_times = {}
# Dictionary to store start times for sitting persons
sitting_start_times = {}

# Initialize plotting data   
XDATA = list(range(PLOT_LENGTH))
YDATA = [0] * PLOT_LENGTH

# Function to start recording
def start_video_recording(frame_size, fps):
    global video_writer
    video_filename = os.path.join(video_folder, f"recorded_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    print(f"Started recording: {video_filename}")

# Function to stop recording
def stop_video_recording():
    global video_writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        print("Stopped recording")

# Function to save the current frame to the video
def save_frame_to_video(frame):
    if video_writer is not None:
        video_writer.write(frame)

def detect_person_and_chair(results, conf_threshold=0.3):
    """Detect persons and chairs using YOLO model results."""
    detected_persons = []
    detected_chairs = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = model.names[cls]
            if label == "person" and conf > conf_threshold:
                detected_persons.append((x1, y1, x2, y2))
            elif label == "chair" and conf > conf_threshold:
                detected_chairs.append((x1, y1, x2, y2))
    return detected_persons, detected_chairs

def is_chair_empty(chair, persons):
    """Check if a chair is empty based on detected persons."""
    chair_x1, chair_y1, chair_x2, chair_y2 = chair
    for person in persons:
        person_x1, person_y1, person_x2, person_y2 = person
        if not (person_x2 < chair_x1 or person_x1 > chair_x2 or 
                person_y2 < chair_y1 or person_y1 > chair_y2):
            return False
    return True

def check_motion(x1, y1, x2, y2, person_id):
    """Check if significant motion is detected for a person."""
    global previous_positions
    motion_detected = False
    if person_id in previous_positions:
        prev_x1, prev_y1, prev_x2, prev_y2 = previous_positions[person_id]
        if abs(x1 - prev_x1) > HAND_MOVEMENT_THRESH or abs(y1 - prev_y1) > HAND_MOVEMENT_THRESH or \
           abs(x2 - prev_x2) > HAND_MOVEMENT_THRESH or abs(y2 - prev_y2) > HAND_MOVEMENT_THRESH:
            motion_detected = True
    previous_positions[person_id] = (x1, y1, x2, y2)
    return motion_detected

def beep_sound():
    """Play a beep sound when cheating is detected."""
    duration = 1000  # milliseconds
    freq = 440  # Hz
    pygame.mixer.init()  # Ensure pygame mixer is initialized
    sound = pygame.mixer.Sound(frequency=freq, size=-16, channels=1, buffer=4096)
    sound.play()
    time.sleep(duration / 1000)  # Convert milliseconds to seconds
    sound.stop()

def detect_head_movement(head_pose):
    """Detect significant head movement."""
    return abs(head_pose['x']) > HEAD_MOVEMENT_THRESH or abs(head_pose['y']) > HEAD_MOVEMENT_THRESH

def detect_neck_movement(neck_pose):
    """Detect significant neck movement."""
    return abs(neck_pose['x']) > NECK_MOVEMENT_THRESH or abs(neck_pose['y']) > NECK_MOVEMENT_THRESH

def infer_activity(head_pose, neck_pose, motion_detected):
    """Infer the activity based on head, neck movement, and motion detection."""
    if abs(head_pose['x']) < 5 and abs(head_pose['y']) < 5 and not motion_detected:
        return "Sitting"
    elif abs(head_pose['x']) > 10 or abs(head_pose['y']) > 10 or motion_detected:
        return "Moving"
    else:
        return "Standing"

def avg(current, previous):
    """Calculate a weighted average to smooth cheat detection."""
    return 0.9 * previous + 0.1 * current

def infer_posture(results):
    """Infer the posture of a detected person."""
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility > 0.5:
                if landmark.y > 0.5:
                    return "Sitting"
                else:
                    return "Standing"
    return "Unknown"

def detect_head_pose(image, face_mesh_results):
    """Detect the head pose using MediaPipe FaceMesh."""
    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []

    for face_landmarks in face_mesh_results.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [33, 263, 1, 61, 291, 199]:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        if face_2d and face_3d:
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x_angle = angles[0] * 360
            y_angle = angles[1] * 360

            return {'x': x_angle, 'y': y_angle}
    return {'x': 0, 'y': 0}

def detect_neck_pose(image, face_mesh_results):
    """Detect the neck pose using MediaPipe FaceMesh."""
    img_h, img_w, _ = image.shape
    neck_3d = []
    neck_2d = []

    for face_landmarks in face_mesh_results.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [78, 308, 14, 13, 152, 148]:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                neck_2d.append([x, y])
                neck_3d.append([x, y, lm.z])

        if neck_2d and neck_3d:
            neck_2d = np.array(neck_2d, dtype=np.float64)
            neck_3d = np.array(neck_3d, dtype=np.float64)
            focal_length = img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(neck_3d, neck_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x_angle = angles[0] * 360
            y_angle = angles[1] * 360

            return {'x': x_angle, 'y': y_angle}
    return {'x': 0, 'y': 0}

def process_cheating(head_pose, neck_pose, motion_detected, frame):
    """Process and detect potential cheating activity."""
    global GLOBAL_CHEAT, PERCENTAGE_CHEAT, CHEAT_THRESH
    head_movement_detected = detect_head_movement(head_pose)
    neck_movement_detected = detect_neck_movement(neck_pose)

    PERCENTAGE_CHEAT = avg(float(head_movement_detected or neck_movement_detected), PERCENTAGE_CHEAT)

    if PERCENTAGE_CHEAT > CHEAT_THRESH:
        GLOBAL_CHEAT = 1
        beep_sound()
    else:
        GLOBAL_CHEAT = 0

    activity = infer_activity(head_pose, neck_pose, motion_detected)
    return activity

# Function to detect mobile phones using the custom YOLO model
def detect_mobile_phones(frame, conf_threshold=MOBILE_CONF_THRESHOLD):
    # Prepare the frame for YOLO input
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    mobile_model.setInput(blob)
    
    # Get the output layer names
    output_layer_names = mobile_model.getUnconnectedOutLayersNames()
    
    # Forward pass to get the output
    layer_outputs = mobile_model.forward(output_layer_names)
    
    height, width = frame.shape[:2]
    detected_mobiles = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]  # Class scores start after the first 5 values
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter by confidence threshold
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, boxW, boxH) = box.astype("int")
                
                x1 = int(centerX - (boxW / 2))
                y1 = int(centerY - (boxH / 2))
                x2 = int(centerX + (boxW / 2))
                y2 = int(centerY + (boxH / 2))
                
                if yolo_classes[class_id] == "cell phone":
                    detected_mobiles.append((x1, y1, x2, y2))
    
    return detected_mobiles

# Function to log mobile usage time
def log_mobile_usage_time(log_file, person_id, start_time, end_time):
    with open(log_file, 'a') as file:
        usage_time = end_time - start_time
        timestamp_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        timestamp_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        file.write(f"Person ID: {person_id}\n")
        file.write(f"Mobile usage from {timestamp_start} to {timestamp_end} - Duration: {usage_time:.2f} seconds\n\n")

def run_detection(source=0):
    global XDATA, YDATA, empty_chair_detection_times, sitting_start_times, video_writer
    
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0, PLOT_LENGTH)
    ax.set_ylim(0, 1)
    line, = ax.plot(XDATA, YDATA, 'r-')
    plt.title("Suspicious Behavior Detection")
    plt.xlabel("Time")
    plt.ylabel("Cheat Probability")

    cap = cv2.VideoCapture(source)
    
    # Set the duration for the run (3 minutes in seconds)
    run_duration = 3 * 60  # 3 minutes
    start_time = time.time()
    end_time = start_time + run_duration
    
    # Start video recording
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_video_recording((frame_width, frame_height), fps)

    # Log file setup
    log_file = 'backend_details.txt'

    while True:
        current_time = time.time()
        if current_time >= end_time:
            print("Run time completed.")
            break
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model(frame)
        detected_persons, detected_chairs = detect_person_and_chair(results)
        detected_mobiles = detect_mobile_phones(frame)

        for idx, (x1, y1, x2, y2) in enumerate(detected_persons):
            person_id = idx + 1  # Unique ID for each detected person
            person_roi = frame[y1:y2, x1:x2]

            head_pose = {'x': 0, 'y': 0}
            neck_pose = {'x': 0, 'y': 0}
            face_mesh_results = face_mesh.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
            if face_mesh_results.multi_face_landmarks:
                head_pose = detect_head_pose(person_roi, face_mesh_results)
                neck_pose = detect_neck_pose(person_roi, face_mesh_results)

            motion_detected = check_motion(x1, y1, x2, y2, person_id)
            activity = process_cheating(head_pose, neck_pose, motion_detected, frame)

            # Determine posture and sitting time
            posture = "Unknown"
            sitting_time = 0
            pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if pose_results.pose_landmarks:
                posture = infer_posture(pose_results)

            if posture == "Sitting":
                if person_id not in sitting_start_times:
                    sitting_start_times[person_id] = current_time
                sitting_time = int(current_time - sitting_start_times[person_id])
            else:
                if person_id in sitting_start_times:
                    del sitting_start_times[person_id]

            # Log details with a human-readable timestamp
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            details = [
                f"Timestamp: {timestamp}",
                f"Person ID: {person_id}",
                f"Activity: {activity}",
                f"Posture: {posture}",
                f"Sitting Time: {sitting_time} sec",
            ]
            log_backend_details(log_file, details)

            # Color the box green for detected persons
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Person {person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Activity: {activity}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Posture: {posture}", (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if posture == "Sitting":
                cv2.putText(frame, f"Time Sitting: {sitting_time} sec", (x1, y2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Log mobile phone usage if detected within person's bounding box
            for mobile in detected_mobiles:
                mobile_x1, mobile_y1, mobile_x2, mobile_y2 = mobile
                if not (mobile_x2 < x1 or mobile_x1 > x2 or mobile_y2 < y1 or mobile_y1 > y2):
                    # Mobile detected in hand of this person
                    log_mobile_usage_time(MOBILE_USAGE_LOG_FILE, person_id, current_time - 5, current_time)

        # Draw rectangles for detected chairs and check if they are empty
        for chair in detected_chairs:
            chair_x1, chair_y1, chair_x2, chair_y2 = chair
            chair_key = (chair_x1, chair_y1, chair_x2, chair_y2)
            if is_chair_empty(chair, detected_persons):
                if chair_key not in empty_chair_detection_times:
                    empty_chair_detection_times[chair_key] = current_time
                time_elapsed = int(current_time - empty_chair_detection_times[chair_key])
                
                if time_elapsed > EMPTY_CHAIR_THRESHOLD:
                    # Display message if the chair has been empty for more than the threshold
                    cv2.rectangle(frame, (chair_x1, chair_y1), (chair_x2, chair_y2), (0,255,255), 2)
                    cv2.putText(frame, "Chair unoccupied", (chair_x1, chair_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    # Display the time for chairs that have been empty less than the threshold
                    cv2.rectangle(frame, (chair_x1, chair_y1), (chair_x2, chair_y2), (255, 255, 0), 2)
                    cv2.putText(frame, "Empty Chair", (chair_x1, chair_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    cv2.putText(frame, f"Time: {time_elapsed} sec", (chair_x1, chair_y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            else:
                if chair_key in empty_chair_detection_times:
                    del empty_chair_detection_times[chair_key]

        XDATA.append(XDATA[-1] + 1)
        XDATA.pop(0)
        YDATA.append(PERCENTAGE_CHEAT)
        YDATA.pop(0)
        line.set_ydata(YDATA)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save the frame to the video
        save_frame_to_video(frame)

        cv2.imshow("Cheating Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    stop_video_recording()  # Ensure the video recording stops after capturing
    cv2.destroyAllWindows()
    # # Plot and save final cheat probability data
    # plt.ioff()
    # plt.savefig('cheat_probability_plot.png')
    # plt.show()

# Function to log backend details
def log_backend_details(log_file, details):
    with open(log_file, 'a') as file:
        file.write("\n".join(details) + "\n\n")

if __name__ == "__main__":
    try:
        source_choice = input("Choose video source - Webcam (w), Video file (v), or IP Camera (i): ").strip().lower()
        if source_choice == 'w':
            run_detection(source=0)
        elif source_choice == 'v':
            video_path = input("Enter the path to the video file: ").strip()
            run_detection(source=video_path)
        elif source_choice == 'i':
            ip_address = input("Enter the IP address of the camera: ").strip()
            username = input("Enter the username: ").strip()
            password = input("Enter the password: ").strip()
            port = input("Enter the port (default 554): ").strip() or "554"
            path = input("Enter the path (usually 'stream' or similar, if unknown leave empty): ").strip()

            rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/{path}"
            run_detection(source=rtsp_url)
        else:
            print("Invalid choice. Please choose 'w' for Webcam, 'v' for Video file, or 'i' for IP Camera.")
    except ImportError as e:
        print(f"An import error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
