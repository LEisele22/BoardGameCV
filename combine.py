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
from onnxtest import nms, load_model, letterbox
import time
from display import add_pieces, board
from operator import add

##Image and variables

# image_path = "metrics/images/IMG-20251207-WA0041.jpg"
image_path = "training/dataset/train/images/pieces12.jpg"
rect_path = "metrics/images/rect41.jpg"
res_path = "metrics/labels/rect41.txt"
# model = "models/best.onnx"
model = "models/best1.onnx"
CLASSES = ['board', 'whitefigure', 'blackfigure', 'emptyspot']
INPUT_SIZE = 640  # same as training
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45

class_names = ['Board', 'Whitefigure', 'Blackfigure', 'Emptyslot']

radius = 1 #radius for circles at intersections
p_radius = 2 #radius for the pieces
origin = (0,0)
step = (5,5)
width = 50
wstep = 10


cam = cv2.VideoCapture(1) #camera

## detecting board

def get_board(frame):
    
    if frame is None:
        print("Image could not be loaded. Check your path.")
        return

    EXPECT = 24

    # --- Detect outer rectangle (paper board) ---
    outer_rect, edges = detect_outer_paper_rect(frame)
    if outer_rect is None:
        print("Could not detect board outline.")
        return 0

    # --- Rectify to square ---
    rect, H = rectify_to_square(frame, outer_rect)
    return rect

def save_im(im, im_path):
    try :
        cv2.imwrite(im_path, im)
        k = cv2.waitKey(30)
        cv2.destroyAllWindows()
    except:
        print('could not save image')
    return im_path

## feeding to CNN


def preprocess(img):
    """Prepare input image."""
    # img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found:")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb, scale, pad_w, pad_h = letterbox(img_rgb, INPUT_SIZE)

    img_input = img_lb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, 0)

    return img, img_input, scale, pad_w, pad_h

def infer(onnx_path, img):
    """Run inference on one image."""
    model = load_model(onnx_path)
    img_original, img_input, scale, pad_w, pad_h = preprocess(img)

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


def store_class(results, res_path, filter = False, lab = False):
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
        if lab : 
            line = str(elt[0]) + ' ' + str(elt[1]) + ' ' + str(elt[2]) + ' ' + str(elt[3]) + ' ' + str(elt[4]) +'\n'
        else :
            line = str(elt[0]) + ' ' + str(elt[1]) + ' ' + str(elt[2]) + ' ' + str(elt[3]) + ' ' + str(elt[4]) +' ' + str(elt[5]) + '\n'
        file.write(line)
    file.close()
    

def filter_nodes_class(res_path, im):
    '''
    function to filter out nodes that were classified multiple times.
    Takes results files, checks for center of labels that are too close 
    and keeps the one with the highest score.
    '''

    dim = np.shape(im)
    file = open(res_path)
    lines = file.read()
    lines = lines.splitlines()
    # print('nb of boxes : ', len(lines))
    lablist = []
    new_lines = []
    for elt in lines : 
        parts = elt.strip().split()
        lab = []
        lab.append(int(parts[0]))
        if lab[0] != 0 :
            x1, y1, x2, y2 = map(float, parts[1:5])
            lab.append(x1)
            lab.append(y1)
            lab.append(x2)
            lab.append(y2)
            lab.append(float(parts[5]))
            lablist.append(lab)
    j =0
    indlist = [] #list of indices that were already compared
    for j in range(len(lablist)):

        if j not in indlist : 
            lab = lablist[j]
            x1, y1, x2, y2 = lab[1:5]
            x_c_ref = (x2+x1)/(2 * dim[1])
            y_c_ref = (y2+y1)/(2 * dim[0])
            for i in range(len(lablist)) :
                if i not in indlist :
                    elt = lablist[i]
                    x1, y1, x2, y2 = elt[1:5]
                    x_c = (x2+x1)/(2 * dim[1])
                    y_c = (y2+y1)/(2 * dim[0])
                    #compare coordinates of centers
                    if x_c_ref- 0.07< x_c < x_c_ref + 0.07 : 
                            if y_c_ref- 0.07 < y_c < y_c_ref + 0.07 :
                                #centers are too close : one and only one label must remain
                                indlist.append(i)
                                if elt[5] > lab[5] : #keep line with the most confidence
                                    lab = elt
            new_lines.append(lab)
 


    file.close()
    # print('nb of boxes : ', len(new_lines))
    store_class(new_lines, res_path, filter = True)



## get list of white and black pieces from txt file of labels
def sort(L, key):
    L.sort(key=key)
    return L

def pieces_list(res_path, lab = False):
    '''
    From the results txt file, 
    gets the list of white figures and black figures with numbered nodes
    these list can then be used as input in display.py
    '''
    file = open(res_path, 'r')
    lines = file.read()
    lines = lines.splitlines()
    #check if right number of nodes
    if len(lines) != 24 :
        print('wrong nb of nodes : ', len(lines))
        return [],[]
    #get the labels in the right order
    lablist = []
    new_lines = []
    for i in range(len(lines)) : 
        parts = lines[i].strip().split()
        lab = []
        lab.append(int(parts[0]))
        x1, y1, x2, y2 = map(float, parts[1:5])
        lab.append(x1)
        lab.append(y1)
        lab.append(x2)
        lab.append(y2)
        if not lab : 
            lab.append(float(parts[5]))
        lablist.append(lab) 
    #order them same as the corners
    lablist.sort(key = lambda elt: (elt[2]+elt[4])/2 , reverse = False)
    # lablist.sort(key = lambda elt: elt[1])
    sortedlab = []
    sortedlab = sort(lablist[0:3], key = lambda elt: (elt[1]+ elt[3])/2) + sort(lablist[3:6],  lambda elt: (elt[1]+ elt[3])/2) + sort(lablist[6:9],  lambda elt: (elt[1]+ elt[3])/2) +sort(lablist[9:15],  lambda elt: (elt[1]+ elt[3])/2) +sort(lablist[15:18],  lambda elt: (elt[1]+ elt[3])/2) +sort(lablist[18:21],  lambda elt: (elt[1]+ elt[3])/2) + sort(lablist[21:24],  lambda elt: (elt[1]+ elt[3])/2)
    
    store_class(sortedlab,res_path, filter= True, lab = lab) 
    #get list of pieces
    w_pieces = []
    b_pieces = []
    empty= []
    for i in range(len(sortedlab)):
        if sortedlab[i][0] == 1 :
            w_pieces.append(i)
        elif sortedlab[i][0] == 2 :
            b_pieces.append(i)
        else : 
            empty.append(i)
    print(w_pieces, b_pieces, empty)
    return w_pieces, b_pieces


##draw result

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


def draw_boxes_num(img, results):
    # color = [(0,255,0), (0,0,255), (255,0,0), (127,0,127)]
    i = 0
    for label in results:
        
        class_name = class_names[int(label[0])]
        color = (70*int(label[0]),0,70*(3-int(label[0])))
        x1,y1,x2,y2 = label[1:5]

        # Rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Label
        txlabel = class_name + ' ' + str(label[5])
        cv2.putText(img, str(i), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
        i += 1
    return img


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
    initsize = np.shape(image)
    if resize :
        try :
            image = cv2.resize(image, None, fx = 0.4, fy = 0.4)
            newsize = np.shape(image)
        except :
            print('NO BOARD')
            return 
    bboxes = []
    labels = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            if resize :
                ratio = (newsize[0]/initsize[0], newsize[1]/initsize[1]) 
                x_c = x_c * ratio[1]
                y_c = y_c * ratio[0]
                w = w * ratio[1]
                h = h * ratio[0]
            bboxes.append([x_c, y_c, w, h])
            labels.append([label, x_c, y_c, w, h, float(parts[5])])
        
    if filtered :
        result = draw_boxes_num(image.copy(), labels)
    else : 
        result = draw_boxes_infer(image.copy(), labels)
    # cv2.imshow("Label Test", result)
    # cv2.waitKey(0)
    return result



##performance metrics


def confusion_matrix(res_path, label_path):
    """
    takes labeled txt file and results (filtered) text file 
    returns the confusion matrix for this file
    """
    conf = [[0 for j in range(3)] for i in range(3)]

    exp_w, exp_b = pieces_list(label_path, lab = True)
    res_w, res_b = pieces_list(res_path)
    exp_empty =[]
    for i in range(24):
        if i not in exp_w :
            if i not in exp_b:
                exp_empty.append(i)
    res_empty =[]
    for i in range(24):
        if i not in res_w :
            if i not in res_b:
                res_empty.append(i)
    #now check if pieces in res are well classified
    for p in res_w :
        if p in exp_w :
            conf[0][0] += 1
        if p in exp_b :
            conf[0][1] += 1
        if p in exp_empty :
            conf[0][2] += 1
    for p in res_b :
        if p in exp_w :
            conf[1][0] += 1
        if p in exp_b :
            conf[1][1] += 1
        if p in exp_empty :
            conf[1][2] += 1
    for p in res_empty :
        if p in exp_w :
            conf[2][0] += 1
        if p in exp_b :
            conf[2][1] += 1
        if p in exp_empty :
            conf[2][2] += 1
    return conf



##main
def main(im, rect_path, res_path):
    frame = cv2.imread(image_path)
    im = get_board(frame)
    im_path = save_im(im, rect_path)
    img, lab = infer(model, im)
    store_class(lab, res_path)
    
    filter_nodes_class(res_path, img)
    res_im = test_yolo_label(rect_path, res_path, class_names, resize = True, filtered=False)
    save_im(res_im, rect_path)
    w_pieces, b_pieces = pieces_list(res_path)
    print(w_pieces, b_pieces)

    axes, corner = board(radius, origin, step, width, wstep)
    add_pieces(axes, corner, w_pieces, b_pieces)

def main_cam(cam,rect_path, res_path):
    ret, frame = cam.read()
    # cv2.imshow('feed', frame)
    # cv2.waitKey(0)
    im = get_board(frame)
    
       
    if type(im) == type(0) :
        print('no board')
        return frame
    else :
          
        im_path = save_im(im, rect_path)
        img, lab = infer(model, im)
        store_class(lab, res_path)
        
        filter_nodes_class(res_path, img)
        res_im = test_yolo_label(rect_path, res_path, class_names, resize = False, filtered=False)
        # cv2.imshow('board', res_im)
        save_im(res_im, rect_path)
        w_pieces, b_pieces = pieces_list(res_path)
        print(w_pieces, b_pieces)
        

        axes, corner = board(radius, origin, step, width, wstep)
        add_pieces(axes, corner, w_pieces, b_pieces)
        return res_im
 

if __name__ == "__main__" :

    rect_path = f"live/im/test.jpg"
    res_path = f"live/label/test.txt"
    # main(image_path, rect_path, res_path)
    while True :
        beg = time.time()
        rect_path = f"live/im/test{round(beg)}.jpg"
        res_path = f"live/label/test{round(beg)}.txt"
        im = main_cam(cam, rect_path, res_path)
        cv2.imshow('board', im)
        k = cv2.waitKey(500)
        if k == 113:
            break
        end=time.time()
        print('process time :', end-beg)
        # time.sleep(1)
    
    cv2.destroyAllWindows()
    cam.release()
    # conf_mat = [[0 for j in range(3)] for i in range(3)]
    # for i in range(56,68):
        # if 
        #     lab_path = f"training/dataset/train/labels/pieces{i}.txt"
        #     res_path = f"metrics/NewModelResults/labels/rect{i}.txt"
        #     mat = confusion_matrix(res_path , lab_path)
        #     # conf_mat = list(map(add, mat, conf_mat))
        #     print(mat)
    # print(conf_mat)

    