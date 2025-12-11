import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# YOLOv5 Letterbox
# ---------------------------------------------------------
def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    scale = new_shape / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)

    pad_w = (new_shape - nw) // 2
    pad_h = (new_shape - nh) // 2

    canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized
    return canvas, scale, pad_w, pad_h


# ---------------------------------------------------------
# Load ONNX model
# ---------------------------------------------------------
def load_model(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------
# Preprocess image for ONNX input
# ---------------------------------------------------------
def preprocess(path, input_size=640):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Image not found: " + path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lb, scale, pad_w, pad_h = letterbox(img_rgb, input_size)

    x = lb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)

    return img, x, scale, pad_w, pad_h


# ---------------------------------------------------------
# Extract figure node points from ONNX YOLO output
# Classes:
#   1 = whitefigure
#   2 = blackfigure
#   3 = emptynode
# ---------------------------------------------------------
def extract_points(onnx_output, scale, pad_w, pad_h, conf_thres=0.3):
    preds = np.squeeze(onnx_output)
    points = []

    for det in preds:
        obj_conf = det[4]
        if obj_conf < conf_thres:
            continue

        class_scores = det[5:]
        cls_id = np.argmax(class_scores)
        cls_conf = class_scores[cls_id]
        final_conf = obj_conf * cls_conf

        if final_conf < conf_thres:
            continue

        if cls_id not in [1, 2, 3]:
            continue

        cx, cy, w, h = det[:4]

        x1 = cx - w/2
        y1 = cy - h/2

        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale

        cx = x1 + w/(2 * scale)
        cy = y1 + h/(2 * scale)

        points.append([cx, cy])

    return np.array(points, dtype=np.float32)


# ---------------------------------------------------------
# Find 4 extreme points (outer square corners)
# ---------------------------------------------------------
def detect_outer_square(points):
    if len(points) < 4:
        return None

    tl = points[np.argmin(points[:, 0] + points[:, 1])]
    tr = points[np.argmin(-points[:, 0] + points[:, 1])]
    br = points[np.argmax(points[:, 0] + points[:, 1])]
    bl = points[np.argmax(-points[:, 0] + points[:, 1])]

    return np.array([tl, tr, br, bl], dtype=np.float32)


# ---------------------------------------------------------
# Helper: distance from point to line segment
# ---------------------------------------------------------
def point_line_distance(p, a, b):
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    line = b - a
    if np.all(line == 0):
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, line) / np.dot(line, line), 0, 1)
    projection = a + t * line
    return np.linalg.norm(p - projection)


# ---------------------------------------------------------
# Detect all figure nodes lying on the outer square edges
# ---------------------------------------------------------
def find_points_on_square_edges(points, square, max_distance=10):
    edges = [
        (square[0], square[1]),  # top edge
        (square[1], square[2]),  # right edge
        (square[2], square[3]),  # bottom edge
        (square[3], square[0])   # left edge
    ]

    edge_points = []

    for p in points:
        for a, b in edges:
            d = point_line_distance(p, a, b)
            if d < max_distance:
                edge_points.append(p)
                break

    return np.array(edge_points, dtype=np.float32)


# ---------------------------------------------------------
# Draw detected square and edge points
# ---------------------------------------------------------
def draw_result(img, square, edge_points):
    out = img.copy()

    # Draw the polygon
    pts = square.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    # Draw the four corners
    for p in square:
        cv2.circle(out, (int(p[0]), int(p[1])), 10, (0, 255, 255), -1)

    # Draw points lying on edges
    for p in edge_points:
        cv2.circle(out, (int(p[0]), int(p[1])), 10, (0, 0, 255), -1)

    return out


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    MODEL_PATH = "models/best1.onnx"
    IMAGE_PATH = "training/dataset/train/images/IMG-20251209-WA0007_jpg.rf.082730109e61cc1a14911c520e6ad6b4.jpg"

    session = load_model(MODEL_PATH)

    img_original, img_input, scale, pad_w, pad_h = preprocess(IMAGE_PATH)

    output = session.run(None, {session.get_inputs()[0].name: img_input})[0]

    points = extract_points(output, scale, pad_w, pad_h)

    if len(points) < 4:
        print("Not enough figure points detected.")
        return

    square = detect_outer_square(points)
    if square is None:
        print("Failed to estimate outer square.")
        return

    edge_points = find_points_on_square_edges(points, square)

    img_result = draw_result(img_original, square, edge_points)

    cv2.imwrite("outer_square_with_edge_points.jpg", img_result)
    print("Saved: outer_square_with_edge_points.jpg")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
