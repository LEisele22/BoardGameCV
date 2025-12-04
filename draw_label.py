import cv2
import numpy as np

#to change : 
class_names = ['Board', 'Whitefigure', 'Blackfigure', 'Emptyslot']
class_int = 2 #choose class that you are about to select on image
image_path = "training/dataset/images/pieces02.jpg"
label_path = "training/dataset/labels/pieces02.txt"

#then run file, on the first image select two corners for each rectangle (last rectangle not working, to be fixed), second image shows labels

def click_event(event, x, y, flags, params):
   
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')

        # put coordinates as text on the image
        cv2.putText(im, f'({x},{y})',(x,y),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # draw point on the image
        cv2.circle(im, (x,y), 3, (0,255,255), -1)
        params.append((x,y))



def get_coord_rect(points):
    x0,y0 = points[0]
    x1,y1 = points[1]

    width  = abs(x1-x0)
    height = abs(y1-y0)
    center1 = width/2 + x0
    center2 = height/2 + y0
    return center1, center2, width, height


def format_label(points, class_int, sizeim):
    file = open(label_path, mode = "a")
    for i in range(0,len(points)-2,2):
        xc,yc,w,h = get_coord_rect(points[i:i+2])
        xc  = xc /sizeim[1]
        w = w/sizeim[1]
        h = h/sizeim[0]
        yc = yc/sizeim[0]
        line = str(class_int) + ' ' + str(xc) +' ' + str(yc)+ ' '+ str(w) + ' ' + str(h) + '\n'
        print(line)
        file.write(line)
    file.close()



def yolo_to_xyxy(box, img_w, img_h):
    x_center, y_center, w, h = box

    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    return x1, y1, x2, y2

def draw_boxes(image, bboxes, labels, class_names):
    img_h, img_w = image.shape[:2]

    for box, label in zip(bboxes, labels):
        x1, y1, x2, y2 = yolo_to_xyxy(box, img_w, img_h)
        class_name = class_names[int(label)]

        # Rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

        # Label
        cv2.putText(image, class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return image








in_count = 0
in_points = []

im = cv2.imread(image_path)
im = cv2.resize(im, None, fx = 0.4, fy = 0.4)
size = np.shape(im)


cv2.namedWindow('Label Test')
cv2.setMouseCallback('Label Test', click_event, in_points)
while True:
   cv2.imshow('Label Test',im)
   k = cv2.waitKey(1) & 0xFF
   if k == 113:
      break
format_label(in_points, class_int, size)
cv2.destroyAllWindows()



def test_yolo_label(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx = 0.4, fy = 0.4)

    bboxes = []
    labels = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            bboxes.append([x_c, y_c, w, h])
            labels.append(label)

    result = draw_boxes(image.copy(), bboxes, labels, class_names)

    cv2.imshow("Label Test", result)
    cv2.waitKey(0)




test_yolo_label(
    image_path,
    label_path,
    class_names
)