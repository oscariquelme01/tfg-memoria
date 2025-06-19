import cv2
from EquirectProcessor import EquirectProcessor, remap_single_view
from coordiante_conversion import yolo_box_to_yaw_pitch
from object_distance import calculate_distance
from util import pick_file_from_directory, replace_immediate_parent
from ultralytics import YOLO


def get_goalkeeper_view(all_detections):
    if not len(all_detections):
        return None

    result = all_detections[0]

    for detection in all_detections:
        # simple calculation: 
        # - 90% weight to how far away the person detected is because we assume the closest to the camera is the goalkeeper
        # - 10% to how confident the model is about the detection  
        distance_score = (1 / result['distance']) * 0.9
        confidence_score = result['conf'] * 0.1

        new_distance_score = (1 / detection['distance']) * 0.9
        new_confidence_score = detection['conf'] * 0.1

        if distance_score + confidence_score < new_distance_score + new_confidence_score:
            result = detection

    return result

# Define the 6 standard view directions (yaw, pitch in degrees)
# The first value (yaw) indexes the horizontal view (360 degrees) 0 -> front, 180 (or -180) -> back and so
# The second value (pitch) indexes the vertical view (180 degrees) 0 -> horizon, -> 90 up, -90 -> down and so 
VIEWS = {
    "right": (90, 0),
    "back": (180, 0),
    "left": (270, 0),
    "front": (0, 0),
    # "bottom": (0, -90),
    # "top": (0, 90)
}

# Configuration
OUTPUT_SIZE = (640, 480)
FOV = 90
USE_GPU = True  # Set to False to use CPU parallelization
SOURCES_DIR = "sources"

# Get input file
input_file = pick_file_from_directory(SOURCES_DIR)
if not input_file:
    print("Failed to get input file")
    exit()

output_file = replace_immediate_parent(input_file, 'edited')

# Initialize video capture
cap = cv2.VideoCapture(input_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    exit()

# Initialize processor and precompute mappings
# TODO: to be fair OUTPUT_SIZE doesn't necesarily need to the equal to the generated perspective images
processor = EquirectProcessor(VIEWS, FOV, OUTPUT_SIZE, USE_GPU)
processor.precompute_mappings(first_frame.shape)

# Initialize YOLO model
# TODO: send this to GPU
model = YOLO('yolov8n.pt')

# Setup output video to the configured OUTPUT_SIZE
output_height, output_width = OUTPUT_SIZE
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (output_width, output_height))

print(f"Processing {total_frames} frames at {fps} FPS...")
print(f"Output dimensions: {output_width}x{output_height}")

# Process video
frame_count = 0
import time
start_time = time.time()

# Default output for the first frames before the person is detected
default_view_name = 'front'
last_output_frame = None
output_frame = None

# Save the yaw and pitch calculated from the previous frame to ensure that there are no re-writes of the same frame
current_yaw, current_pitch = 0,0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame retrieving all the views
    views = processor.process_frame(frame)
    all_detections = []
    
    for view_name, view in views.items(): #type: ignore
        results = model(view, verbose=False)
        for result in results:
            for box in result.boxes:
                if box.cls.item() == 0 and box.conf.item() > 0.75:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    yaw, pitch = yolo_box_to_yaw_pitch(x1, y1, x2, y2, output_width, output_height, VIEWS[view_name][0], VIEWS[view_name][1], FOV)

                    all_detections.append({
                        'yaw': yaw,
                        'pitch': pitch,
                        'conf': box.conf.item(),
                        'distance': calculate_distance(x1, y1, x2, y2, model.names[box.cls.item()]),
                        'view_name': view_name
                    })
                    
    print(all_detections)


    # Generate the detection-centered view
    choosen_deteciton =  get_goalkeeper_view(all_detections)
    if choosen_deteciton is not None:
        current_yaw, current_pitch = choosen_deteciton['yaw'], choosen_deteciton['pitch']
        output_frame = remap_single_view(frame, current_yaw, -current_pitch, FOV, OUTPUT_SIZE)

        # Re-run detection on the new centered view to get correct coordinates
        new_results = model(output_frame, verbose=False)
        for new_result in new_results:
            for new_box in new_result.boxes:
                if new_box.cls.item() == 0 and new_box.conf.item() > 0.75:  # Person detected in new view
                    nx1, ny1, nx2, ny2 = new_box.xyxy[0].cpu().numpy()
                    new_conf = new_box.conf.item()

                    distance = calculate_distance(nx1, ny1, nx2, ny2, model.names[new_box.cls.item()])
                    view_name = choosen_deteciton['view_name']

                    # print(f'Frame {frame_count}: found person in view {view_name} with confidence {new_box.conf.item()}.2f at frame {frame_count} and distance {distance}. Yaw: {current_yaw}deg Pitch: {current_pitch}deg')
                    
                    # Draw bounding box on the new output frame
                    cv2.rectangle(output_frame, 
                                 (int(nx1), int(ny1)), (int(nx2), int(ny2)), 
                                 (0, 255, 0), 2)
                    cv2.putText(output_frame, f'Person {new_conf:.2f} @ {distance}', 
                               (int(nx1), int(ny1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # just a single person per view!
                    break


        if not len(new_results):
            print(f'Frame {frame_count}: no results found in the new view created defaulting to yaw {current_yaw}deg & pitch {current_pitch}deg')
        else:
            current_view = choosen_deteciton['view_name']
            print(f'Frame {frame_count}: new view created with detection in {current_view} view at yaw {current_yaw}deg & pitch {current_pitch}deg')
    
    # If no person detected, use the default view
    else:
        output_frame = remap_single_view(frame, current_yaw, -current_pitch) if last_output_frame is not None else views[default_view_name] #type: ignore
        
        no_person_detected_debug_str = f'last output frame with yaw {current_yaw}deg & pitch {current_pitch}deg' if last_output_frame is not None else f'default output frame: {default_view_name}'
        print(f'Frame {frame_count}: no person detected, defaulting to ' + no_person_detected_debug_str)
    
    # Always write exactly one perspective frame
    out.write(output_frame)
    last_output_frame = output_frame
    
    frame_count += 1
    
    # Progress indicator
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        progress = (frame_count / total_frames) * 100
        eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
              f"FPS: {fps_actual:.1f} | ETA: {eta:.1f}s")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
print(f"\nCompleted in {total_time:.2f} seconds")
print(f"Average FPS: {frame_count / total_time:.2f}")
