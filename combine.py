import cv2
import numpy as np
from morris_node_system import (
    detect_outer_paper_rect,
    rectify_to_square,
    detect_nodes,
    sort_nodes_morris_grid,
    draw_graph,
    detect_all_stones
)
from onnxtest import nms, load_model, preprocess


##Image and variables

image_path = "metrics/images/IMG-20251207-WA0046.jpg"
rect_path = "metrics/images/rect46.jpg"
model = "models/best.onnx"
CLASSES = ['board', 'whitefigure', 'blackfigure', 'emptyspot']
INPUT_SIZE = 640  # same as training
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45

class_names = ['Board', 'Whitefigure', 'Blackfigure', 'Emptyslot']

res_path = "metrics/labels/rect46.txt"
## detecting board


def infer(onnx_path, img_path):
    """Run inference on one image."""
    model = load_model(onnx_path)
    img_original, img_input, scale, pad_w, pad_h = preprocess(img_path)

    output = model.run(None, {model.get_inputs()[0].name: img_input})[0]
    preds = np.squeeze(output)

    boxes = []
    scores = []
    class_ids = []
    labels = []
    for det in preds:
        conf = det[4]
        if conf < CONF_THRESHOLD:
            continue

        cls_scores = det[5:]
        cls_id = np.argmax(cls_scores)
        cls_conf = cls_scores[cls_id]

        final_conf = conf * cls_conf
        if final_conf < CONF_THRESHOLD:
            continue

        # YOLOv5 format: x_center, y_center, width, height
        cx, cy, w, h = det[:4]
        # sizeim = np.shape(img_input)
        # xc  = cx /sizeim[1]
        # wb = w/sizeim[1]
        # hb = h/sizeim[0]
        # yc = cy/sizeim[0]
        # # xc  = (cx - pad_w) / scale
        # # wb = (w - pad_w) / scale
        # # hb =(h - pad_h) / scale
        # # yc = (cy - pad_h) / scale
        # labels.append([cls_id, xc, yc, wb, hb, cls_conf])
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        # Reverse padding & scaling
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        labels.append([cls_id, x1, y1, x2, y2, cls_conf])
        boxes.append([x1, y1, x2, y2])
        scores.append(final_conf)
        class_ids.append(cls_id)

    if len(boxes) == 0:
        print("No detections.")
        return img_original, []

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    keep = nms(boxes, scores, IOU_THRESHOLD)

    results = []
    for i in keep:
        results.append({
            "box": boxes[i].tolist(),
            "score": float(scores[i]),
            "class": CLASSES[class_ids[i]]
        })

    return img_original, labels


def get_board(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Image could not be loaded. Check your path.")
        return

    EXPECT = 24

    # --- Detect outer rectangle (paper board) ---
    outer_rect, edges = detect_outer_paper_rect(frame)
    if outer_rect is None:
        print("Could not detect board outline.")
        return 0,None, None

    # --- Rectify to square ---
    rect, H = rectify_to_square(frame, outer_rect)
    return rect

def save_im(im, im_path):
    cv2.imwrite(im_path, im)
    k = cv2.waitKey(30)
    cv2.destroyAllWindows()
    return im_path

##feeding board to CNN



def store_class(results, res_path, filter = False):
    '''
    stores result of the CNN run on the rectified board 
    (class, coordinates of rectangle and confidence)
    '''
    try :
        file = open(res_path, mode = 'x')
    except FileExistsError :
        if filter :
            file = open(res_path, mode = 'w')
        else :
            file = open(res_path, mode = 'a')
    for elt in results : 
        line = str(elt[0]) + ' ' + str(elt[1]) + ' ' + str(elt[2]) + ' ' + str(elt[3]) + ' ' + str(elt[4]) +' ' + str(elt[5]) + '\n'
        file.write(line)
    file.close()
    

def filter_nodes_class(res_path, im):
    '''
    function to filter out nodes that were classified multiple times.
    Takes results files, checks for center of labels that are too close 
    and keeps the one with the highest score.
    '''

    dim = np.shape(img)
    file = open(res_path)
    lines = file.read()
    lines = lines.splitlines()
    print('nb of boxes : ', len(lines))
    lablist = []
    new_lines = []
    for i in range(len(lines)) : 
        parts = lines[i].strip().split()
        lab = []
        lab.append(int(parts[0]))
        x1, y1, x2, y2 = map(float, parts[1:5])

        # x_c = (x2-x1)/2
        # y_c = (y2-y1)/2
        # w = x2-x1
        # h = y2-y1

        # lab.append(x_c)
        # lab.append(y_c)
        # lab.append(w)
        # lab.append(h)
        lab.append(x1)
        lab.append(y1)
        lab.append(x2)
        lab.append(y2)
        lab.append(float(parts[5]))
        lablist.append(lab)
    j =0
    indlist = [] #list of indices that were already compared
    while j < len(lablist):
        if j not in indlist : 
            lab = lablist[j]
            x1, y1, x2, y2 = lab[1:5]
            x_c_ref = (x2-x1)/(2)
            y_c_ref = (y2-y1)/(2)
            for i in range(len(lablist)) :
                if i not in indlist :
                    elt = lablist[i]
                    x1, y1, x2, y2 = elt[1:5]
                    x_c = (x2-x1)/(2)
                    y_c = (y2-y1)/(2)
                    #compare coordinates of centers
                    if x_c_ref- 0.5 < x_c < x_c_ref + 0.5 : 
                            if y_c_ref- 0.5 < y_c < y_c_ref + 0.5 :
                                #centers are too close : one and only one label must remain
                                indlist.append(i)
                                if elt[5] > lab[5] : #keep line with the most confidence
                                    lab = elt
            new_lines.append(lab)
        j += 1


    file.close()
    print('nb of boxes : ', len(new_lines))
    # form_lines = []
    # for elt in new_lines :
    #     x_c, y_c, w, h = elt[1:5]
    #     x2 = x_c - w/2
    #     x1 = x_c + w/2
    #     y2 = y_c - h/2
    #     y1 = y_c + h/2

    #     form_lines.append([elt[0], x1, y1, x2, y2, elt[5]])
    store_class(new_lines, res_path, filter = True)

def pieces_list(res_path):
    '''
    From the results txt file, 
    gets the list of white figures and black figures with numbered nodes
    these list can then be used as input in display.py
    '''


def draw_boxes_infer(img, results):
    """Draw boxes on the image."""
    # color = [(0,255,0), (0,0,255), (255,0,0), (127,0,127)]
    for label in results:
        class_name = class_names[int(label[0])]
        color = (70*int(label[0]),0,70*(3-int(label[0])))
        x1,y1,x2,y2 = label[1:5]

        # Rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Label
        txlabel = class_name + ' ' + str(label[5])
        cv2.putText(img, class_name, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
    return img


def draw_boxes_filtered(image, labels):
    img_h, img_w = image.shape[:2]

    for label in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(label[1:5], img_w, img_h)
        class_name = class_names[int(label[0])]
        color = (70*int(label[0]),0,70*(3-int(label[0])))
        # Rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label
        cv2.putText(image, class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image


def yolo_to_xyxy(box, img_w, img_h):
    x_center, y_center, w, h = box

    # x_center *= img_w
    # y_center *= img_h
    # w *= img_w
    # h *= img_h

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    return x1, y1, x2, y2


def test_yolo_label(image_path, label_path, class_names, resize, filtered = False):
    image = cv2.imread(image_path)
    if resize :
        image = cv2.resize(image, None, fx = 0.4, fy = 0.4)

    bboxes = []
    labels = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            bboxes.append([x_c, y_c, w, h])
            labels.append([label, x_c, y_c, w, h, float(parts[5])])
        
    if filtered :
        result = draw_boxes_filtered(image.copy(), labels)
    else : 
        result = draw_boxes_infer(image.copy(), labels)
    cv2.imshow("Label Test", result)
    cv2.waitKey(0)

##display output

if __name__ == "__main__" :

    im = get_board(image_path)
    im = cv2.resize(im, None, fx = 0.4, fy = 0.4)
    title = f"rectified board"
    cv2.namedWindow(title)
    while True:
        cv2.imshow(title,im)
        k = cv2.waitKey(30)
        if k == 113:
            break
    cv2.destroyAllWindows()
    im_path = save_im(im, rect_path)
    img, lab = infer(model, rect_path)
    print(lab)
    store_class(lab, res_path)
    test_yolo_label(rect_path, res_path, class_names, resize = False, filtered=False)
    filter_nodes_class(res_path, img)
    test_yolo_label(rect_path, res_path, class_names, resize = False, filtered=False)
    # title = f"pieces detection"
    # cv2.namedWindow(title)
    # while True:
    #     cv2.imshow(title,im)
    #     k = cv2.waitKey(30)
    #     if k == 113:
    #         break
    # cv2.destroyAllWindows()


    