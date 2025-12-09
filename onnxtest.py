import cv2
import numpy as np
import onnxruntime as ort

# ------------- CONFIG ----------------
CLASSES = ['board', 'whitefigure', 'blackfigure', 'emptyspot']
INPUT_SIZE = 640  # same as training
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
# -------------------------------------

def letterbox(img, new_shape=640):
    """Resize image with unchanged aspect ratio and padding."""
    h, w = img.shape[:2]
    scale = new_shape / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    
    image_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    canvas[(new_shape - nh)//2 : (new_shape - nh)//2 + nh,
           (new_shape - nw)//2 : (new_shape - nw)//2 + nw] = image_resized
    
    return canvas, scale, (new_shape - nw)//2, (new_shape - nh)//2


def load_model(onnx_path):
    """Load ONNX model with ONNX Runtime."""
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def preprocess(img_path):
    """Prepare input image."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb, scale, pad_w, pad_h = letterbox(img_rgb, INPUT_SIZE)

    img_input = img_lb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, 0)

    return img, img_input, scale, pad_w, pad_h


def nms(boxes, scores, iou_thres):
    """Non-Maximum Suppression."""
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        xx1 = np.maximum(boxes[i][0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i][1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i][2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i][3], boxes[idxs[1:], 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = ((boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1])
                 + (boxes[idxs[1:], 2]-boxes[idxs[1:], 0]) * 
                   (boxes[idxs[1:], 3]-boxes[idxs[1:], 1])
                 - inter)

        iou = inter / (union + 1e-6)
        idxs = idxs[1:][iou < iou_thres]

    return keep


def infer(onnx_path, img_path):
    """Run inference on one image."""
    model = load_model(onnx_path)
    img_original, img_input, scale, pad_w, pad_h = preprocess(img_path)

    output = model.run(None, {model.get_inputs()[0].name: img_input})[0]
    preds = np.squeeze(output)

    boxes = []
    scores = []
    class_ids = []

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

        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        # Reverse padding & scaling
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

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

    return img_original, results


def draw_boxes(img, results):
    """Draw boxes on the image."""
    for r in results:
        x1, y1, x2, y2 = map(int, r["box"])
        label = f"{r['class']} {r['score']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)
    return img


onnx_model = "models/best1.onnx"
img_path = "training/dataset/valid/images/test02.jpg"

img, results = infer(onnx_model, img_path)

print("Detections:", results)

img_drawn = draw_boxes(img.copy(), results)
cv2.imwrite("result.jpg", img_drawn)
