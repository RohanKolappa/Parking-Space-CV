import cv2
import pickle
import numpy as np

# Bounding Boxes for regular spaces and disabled spaces
rectangles = []
d_rectangles = []

# Width and height of parking space
delta_x = 69
delta_y = 30

# Width and height of disabled parking space
d_delta_x = 64
d_delta_y = 35

# Used to adjust bounding box placement due to camera movement
offset = 0

# Counter for saving cropped images for training
j = 0

# Load Bounding Boxes
try:
    with open('./Bounding Boxes/rectangles2.pkl', 'rb') as file:
        rectangles = pickle.load(file)
        d_rectangles = pickle.load(file)
except:
    print(f'rectangles2.pkl does not exist')

num_spaces = len(rectangles)
num_d_spaces = len(d_rectangles)
    
video_path = '.\data\parking_1920_1080.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps)
frame_count = 0
    
    
while True:
    ret, frame = cap.read()
    
    # Grayscale and convert frame to binary image
    binary_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary_frame = cv2.adaptiveThreshold(src=binary_frame, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         thresholdType=cv2.THRESH_BINARY_INV, blockSize=53, C=30)
    
    if not ret:
        print("Reached the end of the video or there is an error.")
        break
    
    # Wait 25 ms per frame, exits if user press Q
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    # Adjust the bounding boxes due to the camera movement
    seconds = (frame_count / interval)
    if 10 <= seconds <= 14:
        offset = int(8 * (seconds - 10) / (14 - 10))
    if 47 <= seconds <= 51:
        offset = int(8 * (51 - seconds) / (51 - 47))
    if seconds == 56:
        frame_count = 0
        
    # Save cropped bounding boxes as images every 2 seconds for training
    # if frame_count % 60 == 0:
    #     for r in rectangles:
    #         f_name = os.path.join(output_dir, f'frame_{j}.jpg')
    #         cv2.imwrite(f_name, frame[r[1] : r[1] + delta_y, r[0] + offset : r[0] + delta_x + offset])
    #         j += 1
        
    #     for r in d_rectangles:
    #         f_name = os.path.join(output_dir, f'frame_{j}.jpg')
    #         cv2.imwrite(f_name, frame[r[1] : r[1] + d_delta_y, r[0] + offset : r[0] + d_delta_x + offset])
    #         j += 1
    
    num_available = 0
    num_d_available = 0
    
    # Draw regular parking bounding boxes
    for r in rectangles:
        w_pixel_count = np.sum(binary_frame[r[1] : r[1] + delta_y, r[0] + offset : r[0] + delta_x + offset] == 255)
        
        # Occupied Parking Space - Red
        color = (0, 0, 255)
        if w_pixel_count < 250:
            # Empty Parking Space - Green
            color = (0, 255, 0)
            num_available += 1

        cv2.rectangle(img=frame, pt1=(r[0] + offset, r[1]), pt2=(r[0] + delta_x + offset, r[1] + delta_y), 
                      color=color, thickness=2)
        cv2.putText(img=frame, text=str(w_pixel_count), org=(r[0] + 5, r[1] + 25), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=0.4, color=(255, 255, 255), thickness=1)
    
    # Draw disabled parking bounding boxes
    for r in d_rectangles:
        w_pixel_count = np.sum(binary_frame[r[1] : r[1] + delta_y, r[0] + offset : r[0] + delta_x + offset] == 255)
        
        # Occupied Disabled Parking Space - Red
        color = (255, 0, 255)
        if w_pixel_count < 250:
            # Empty Disabled Parking Space - Blue
            color = (255, 0, 0)
            num_d_available += 1
        
        cv2.rectangle(img=frame, pt1=(r[0] + offset, r[1]), pt2=(r[0] + d_delta_x + offset, r[1] + d_delta_y), 
                      color=color, thickness=2)
        cv2.putText(img=frame, text=str(w_pixel_count), org=(r[0] + 5, r[1] + 25), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=0.4, color=(255, 255, 255), thickness=1)
    
    # Update available parking counter
    cv2.putText(img=frame, text=f'Available Spaces: {num_available}/{num_spaces}', org=(30,30), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)
    cv2.putText(img=frame, text=f'Available Disabled Spaces: {num_d_available}/{num_d_spaces}', org=(30,60), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)
    
    cv2.imshow('Parking Space Detector', frame)
    
    # Uncomment this to draw the binary image version of the frame
    # cv2.imshow('Parking Lot', binary_frame)
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()