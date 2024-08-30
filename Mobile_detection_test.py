import time
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Constants for mobile detection
MOBILE_CONF_THRESHOLD = 0.5
MOBILE_USAGE_LOG_FILE = 'mobile_usage_log.txt'

# Load the custom YOLO model for mobile detection
# mobile_model = YOLO('custom_yolov8_mobile.pt')  # Replace with your custom model file

# Load YOLOv3 model
mobile_model = cv2.dnn.readNet("yolov3W.weights", "yolov3.cfg")

# Load class labels
yolo_classes = []
with open('coco.names', 'r') as f:
    yolo_classes = f.read().splitlines()

# Load the existing YOLO model for persons and chairs
model = YOLO('yolov8x.pt')  # This model should include 'person' and 'chair' classes
    

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


# Function to detect mobile phones using the custom model
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

def run_detection_with_mobile(source=0):
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Detect persons and chairs
        results = model(frame)
        detected_persons, detected_chairs = detect_person_and_chair(results)
        
        # Detect mobile phones
        detected_mobiles = detect_mobile_phones(frame)
        
        # Log mobile usage
        current_time = time.time()
        for idx, (x1, y1, x2, y2) in enumerate(detected_persons):
            person_id = idx + 1  # Unique ID for each detected person
            for mobile in detected_mobiles:
                mobile_x1, mobile_y1, mobile_x2, mobile_y2 = mobile
                if not (mobile_x2 < x1 or mobile_x1 > x2 or mobile_y2 < y1 or mobile_y1 > y2):
                    # Mobile detected in hand of this person
                    log_mobile_usage_time(MOBILE_USAGE_LOG_FILE, person_id, current_time - 5, current_time)
        
        # Display the frame with detections (this part is optional and can be removed if not needed)
        for (x1, y1, x2, y2) in detected_persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        for (mobile_x1, mobile_y1, mobile_x2, mobile_y2) in detected_mobiles:
            cv2.rectangle(frame, (mobile_x1, mobile_y1), (mobile_x2, mobile_y2), (255, 0, 0), 2)
            cv2.putText(frame, "Mobile", (mobile_x1, mobile_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow("Detection with Mobile", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        source_choice = input("Choose video source - Webcam (w), Video file (v), or IP Camera (i): ").strip().lower()
        if source_choice == 'w':
            run_detection_with_mobile(source=0)
        elif source_choice == 'v':
            video_path = input("Enter the path to the video file: ").strip()
            run_detection_with_mobile(source=video_path)
        elif source_choice == 'i':
            ip_address = input("Enter the IP address of the camera: ").strip()
            username = input("Enter the username: ").strip()
            password = input("Enter the password: ").strip()
            port = input("Enter the port (default 554): ").strip() or "554"
            path = input("Enter the path (usually 'stream' or similar, if unknown leave empty): ").strip()

            rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/{path}"
            run_detection_with_mobile(source=rtsp_url)
        else:
            print("Invalid choice. Please choose 'w' for Webcam, 'v' for Video file, or 'i' for IP Camera.")
    except ImportError as e:
        print(f"An import error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")