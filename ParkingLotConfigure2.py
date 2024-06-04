import cv2
import pickle

window_name = 'Parking Lot'

delta_x = 69
delta_y = 30

d_delta_x = 64
d_delta_y = 35

rectangles = [(151,152)]
d_rectangles = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangles.append((x,y))
        
    if event == cv2.EVENT_RBUTTONDOWN:
        for i, val in enumerate(rectangles):
            if val[0] <= x <= val[0] + delta_x and val[1] <= y <= val[1] + delta_y:
                rectangles.pop(i)
        
        for i, val in enumerate(d_rectangles):
            if val[0] <= x <= val[0] + d_delta_x and val[1] <= y <= val[1] + d_delta_y:
                d_rectangles.pop(i)
    
    if event == cv2.EVENT_MBUTTONDOWN:
        d_rectangles.append((x,y))

try:
    with open('./Bounding Boxes/rectangles2.pkl', 'rb') as file:
        rectangles = pickle.load(file)
        d_rectangles = pickle.load(file)
except:
    print(f'rectangles2.pkl does not exist')

while True:
    lot_img = cv2.imread('./Data/baseParkingImgFull.jpg')

    for r in rectangles:
        cv2.rectangle(lot_img, (r[0], r[1]), (r[0] + delta_x, r[1] + delta_y), (0, 0, 255), 1)
    
    for r in d_rectangles:
        cv2.rectangle(lot_img, (r[0], r[1]), (r[0] + d_delta_x, r[1] + d_delta_y), (255, 0, 0), 1)
        
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 2016, 1020)
    cv2.imshow(window_name, lot_img)
    cv2.setMouseCallback(window_name, mouse_callback)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        with open('./Bounding Boxes/rectangles2.pkl', 'wb') as file:
            pickle.dump(rectangles, file)
            pickle.dump(d_rectangles, file)
        break
    