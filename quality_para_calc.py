import cv2
import numpy as np
import onnxruntime as ort
import os
from glob import glob
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ------------- CONFIG ----------------
CLASSES = ['board', 'whitefigure', 'blackfigure', 'emptyspot']
INPUT_SIZE = 640
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
MATCH_IOU = 0.5        # IOU für GT–Prediction Matching
# -------------------------------------

def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    scale = new_shape / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    image_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)

    pad_h = (new_shape - nh) // 2
    pad_w = (new_shape - nw) // 2

    canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = image_resized
    return canvas, scale, pad_w, pad_h

def load_model(onnx_path):
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def preprocess(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb, scale, pad_w, pad_h = letterbox(img_rgb, INPUT_SIZE)

    img_input = img_lb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, 0)
    return img, img_input, scale, pad_w, pad_h

def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]

        xx1 = np.maximum(boxes[i][0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i][1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i][2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i][3], boxes[rest, 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area1 = (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1])
        area2 = (boxes[rest, 2]-boxes[rest, 0]) * (boxes[rest, 3]-boxes[rest, 1])
        union = area1 + area2 - inter
        iou = inter / (union + 1e-6)

        idxs = rest[iou < iou_thres]

    return keep

def infer_single(model, img_path):
    img_original, img_input, scale, pad_w, pad_h = preprocess(img_path)
    preds = model.run(None, {model.get_inputs()[0].name: img_input})[0]
    preds = np.squeeze(preds)

    boxes, scores, class_ids = [], [], []

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

        cx, cy, w, h = det[:4]
        x1 = (cx - w/2 - pad_w) / scale
        y1 = (cy - h/2 - pad_h) / scale
        x2 = (cx + w/2 - pad_w) / scale
        y2 = (cy + h/2 - pad_h) / scale

        boxes.append([x1, y1, x2, y2])
        scores.append(final_conf)
        class_ids.append(cls_id)

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    keep = nms(boxes, scores, IOU_THRESHOLD)
    results = [{"box": boxes[i], "score": float(scores[i]), "class_id": int(class_ids[i])} for i in keep]
    return results

def load_ground_truth(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, cx, cy, w, h = map(float, line.split())
            cls = int(cls)

            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h

            labels.append({"class": cls, "box": np.array([x1, y1, x2, y2])})

    return labels
# --- Deine bisherigen Imports & Code bleiben unverändert bis load_ground_truth() ---

def load_ground_truth(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, cx, cy, w, h = map(float, line.split())
            cls = int(cls)

            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h

            labels.append({"class": cls, "box": np.array([x1, y1, x2, y2])})
    return labels


# -----------------------------------------------------------
# NEUE FUNKTION: filtert die board-Klasse (class_id = 0) raus
# -----------------------------------------------------------
def filter_without_board(preds):
    return [p for p in preds if p["class_id"] != 0]


# -----------------------------------------------------------
# OPTIONAL: infer_single erweitert, um Filtering direkt zu nutzen
# -----------------------------------------------------------
def infer_single(model, img_path, ignore_board=False):
    img_original, img_input, scale, pad_w, pad_h = preprocess(img_path)
    preds = model.run(None, {model.get_inputs()[0].name: img_input})[0]
    preds = np.squeeze(preds)

    boxes, scores, class_ids = [], [], []

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

        cx, cy, w, h = det[:4]
        x1 = (cx - w/2 - pad_w) / scale
        y1 = (cy - h/2 - pad_h) / scale
        x2 = (cx + w/2 - pad_w) / scale
        y2 = (cy + h/2 - pad_h) / scale

        boxes.append([x1, y1, x2, y2])
        scores.append(final_conf)
        class_ids.append(cls_id)

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    keep = nms(boxes, scores, IOU_THRESHOLD)
    results = [{"box": boxes[i], "score": float(scores[i]), "class_id": int(class_ids[i])} for i in keep]

    # ---- Neu: automatisch board ignorieren, wenn gewünscht ----
    if ignore_board:
        results = filter_without_board(results)

    return results


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def evaluate_dataset(model_path, valid_images_folder):
    model = load_model(model_path)
    image_paths = glob(os.path.join(valid_images_folder, "*.jpg"))

    y_true_all = []
    y_pred_all = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

        gt_items = load_ground_truth(label_path, img_w, img_h)
        detections = infer_single(model, img_path)
        
        # TRUE POSITIVES & FALSE POSITIVES
        for det in detections:
            matched = False
            for gt in gt_items:
                if det["class_id"] == gt["class"] and box_iou(det["box"], gt["box"]) >= MATCH_IOU:
                    y_true_all.append(gt["class"])
                    y_pred_all.append(det["class_id"])
                    matched = True
                    break

            if not matched:
                y_true_all.append(-1)
                y_pred_all.append(det["class_id"])

        # FALSE NEGATIVES
        for gt in gt_items:
            matched = any(
                det["class_id"] == gt["class"] and box_iou(det["box"], gt["box"]) >= MATCH_IOU
                for det in detections
            )
            if not matched:
                y_true_all.append(gt["class"])
                y_pred_all.append(-1)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    print("\n===== MODEL EVALUATION =====")
    print("Precision:", precision_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    print("Recall:", recall_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    print("F1 Score:", f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))

    print("\nConfusion Matrix (-1 = FP/FN cases):")
    print(confusion_matrix(y_true_all, y_pred_all))

    return

# -------------------- RUN EVALUATION --------------------
onnx_model = "models/best1.onnx"
valid_images_folder = "training/dataset/valid/images"

evaluate_dataset(onnx_model, valid_images_folder)
