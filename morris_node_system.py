import cv2
import numpy as np
import math

# =========================
# Basic geometry helpers
# =========================
def sample_patch(img, center, r=12):
    x, y = int(center[0]), int(center[1])
    return img[max(0,y-r):y+r, max(0,x-r):x+r]

def classify_stone(patch):
    if patch.size == 0:
        return "empty"

    Z = patch.reshape(-1,3).astype(np.float32)

    # kmeans for background vs object
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 15, 1)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    c0, c1 = centers
    b0 = np.mean(c0)  # brightness
    b1 = np.mean(c1)

    darker = 0 if b0 < b1 else 1
    darker_fraction = np.mean(labels == darker)

    if darker_fraction < 0.10:
        return "empty"
    return "stone"

def stone_color(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    m = np.mean(gray)

    # adjust threshold based on your lighting
    return "black" if m < 120 else "white"

def detect_all_stones(rectified, full_nodes):
    states = []

    for (x,y) in full_nodes:
        patch = sample_patch(rectified, (x,y))
        status = classify_stone(patch)

        if status == "empty":
            states.append("empty")
        else:
            states.append(stone_color(patch))

    return states

def sort_nodes_morris_grid(nodes):
    row_sizes = [3,3,3,6,3,3,3]  # THIS is your layout
    
    pts = np.array(nodes, dtype=np.float32)
    ptsY = pts[np.argsort(pts[:,1])]  # sort by Y
    
    rows = []
    idx = 0
    for size in row_sizes:
        row = ptsY[idx:idx+size]
        row = row[np.argsort(row[:,0])]  # sort row by X
        rows.append(row)
        idx += size
    
    return np.vstack(rows)


def order_points_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
    idx = np.argsort(angles)
    return pts[idx]

def detect_outer_paper_rect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    H, W = frame.shape[:2]
    img_area = H * W

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            ar = cv2.contourArea(approx)
            if 0.2*img_area < ar < 0.98*img_area:
                if ar > best_area:
                    best_area = ar
                    best = approx.reshape(4,2)

    if best is None:
        return None, edges

    # roughly order corners
    s = best.sum(axis=1)
    tl = best[np.argmin(s)]
    br = best[np.argmax(s)]
    diff = np.diff(best, axis=1).reshape(-1)
    tr = best[np.argmin(diff)]
    bl = best[np.argmax(diff)]
    ordered = np.array([tl,tr,br,bl], dtype=np.float32)

    return ordered, edges

def rectify_to_square(frame, corners):
    (tl, tr, br, bl) = corners
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    size = int(max(wA, wB, hA, hB))
    size = max(size, 400)

    dst = np.array([[0,0],
                    [size-1, 0],
                    [size-1, size-1],
                    [0, size-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    warp = cv2.warpPerspective(frame, H, (size, size))
    return warp, H

# =========================
# Node detection (black dots)
# =========================

def detect_nodes(rect):
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    # Invert so dots become white-ish blobs
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean small noise
    bw = cv2.medianBlur(bw, 5)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 2000:
            continue
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        circ = 4*math.pi*area / (cv2.arcLength(cnt, True)**2 + 1e-6)
        # roughly circular
        if circ < 0.5:
            continue
        centers.append((x,y))

    centers = np.array(centers, dtype=np.float32)
    return centers

def sort_nodes_morris_grid(nodes):
    """
    Sort the 24 detected nodes into Morris board order:
    Y-sorted into 7 rows with node counts:
        [3, 3, 3, 6, 3, 3, 3]
    Within each row: X-sorted left->right
    Returns: (24,2)
    """
    # exact Morris row structure
    row_sizes = [3, 3, 3, 6, 3, 3, 3]

    pts = np.array(nodes, dtype=np.float32)
    
    # Sort everything by Y first
    ptsY = pts[np.argsort(pts[:,1])]

    # Split into the exact row sizes
    rows = []
    index = 0
    for size in row_sizes:
        row = ptsY[index : index + size]
        # sort row left -> right
        row = row[np.argsort(row[:,0])]
        rows.append(row)
        index += size

    return np.vstack(rows)



# Morris connectivity
edges = [
    # outer square
    (0,1),(1,2),(2,23),(23,21),(21,0),

    # mid square
    (3,4),(4,5),(5,20),(20,18),(18,3),

    # inner square
    (6,7),(7,8),(8,17),(17,15),(15,6),

    # connections between squares (spokes)
    (0,3),(1,4),(2,5),
    (3,6),(4,7),(5,8),
    (21,18),(22,19),(23,20),
    (18,15),(19,16),(20,17),

    # middle long row (the 6 nodes)
    (9,10),(10,11),(12,13),(13,14),

    # connections to long middle row
    
]


def draw_graph(rect_disp, full_nodes):
    # draw nodes
    for idx,(x,y) in enumerate(full_nodes):
        cv2.circle(rect_disp, (int(x),int(y)), 6, (0,255,255), -1)
        cv2.putText(rect_disp, str(idx), (int(x)+6,int(y)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # draw edges
    for a,b in edges:
        x1,y1 = full_nodes[a]
        x2,y2 = full_nodes[b]
        cv2.line(rect_disp, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,255), 2)

    return rect_disp


# =========================
# Main
# =========================

def main():
    saved_nodes = None

    cap = cv2.VideoCapture(2)
    EXPECT = 24

    while True:
        ret, frame = cap.read()
        if not ret:
            print("I hate my life")
            break

        outer_rect, edges = detect_outer_paper_rect(frame)
        disp = frame.copy()

        if outer_rect is not None:
            for (x,y) in outer_rect:
                cv2.circle(disp, (int(x),int(y)), 5, (0,255,0), -1)

            rect, H = rectify_to_square(frame, outer_rect)
            cv2.imshow("Rectified raw", rect)

            if saved_nodes is None:
                # First frame: detect nodes normally
                nodes = detect_nodes(rect)
                if len(nodes) == 24:
                    saved_nodes = sort_nodes_morris_grid(nodes)
                    print("Node map initialized.")
                else:
                    print("Waiting for clean board to initialize node map...")
                    continue
            else:
                # Use frozen nodes
                full_nodes = saved_nodes
            debug_nodes = rect.copy()
            for (x,y) in nodes:
                cv2.circle(debug_nodes, (int(x),int(y)), 4, (0,255,255), -1)
            cv2.imshow("Detected nodes", debug_nodes)

            if len(nodes) == EXPECT:
                full_nodes = sort_nodes_morris_grid(nodes)

                # draw graph
                rect_graph = rect.copy()
                rect_graph = draw_graph(rect_graph, full_nodes)

                # detect stones
                stone_states = detect_all_stones(rect, full_nodes)
                # overlay stones
                for state, (x,y) in zip(stone_states, full_nodes):
                    if state == "black":
                        cv2.circle(rect_graph, (int(x),int(y)), 10, (0,0,0), -1)
                    elif state == "white":
                        cv2.circle(rect_graph, (int(x),int(y)), 10, (255,255,255), -1)
                    else:
                        cv2.circle(rect_graph, (int(x),int(y)), 6, (0,255,0), 2)

                cv2.imshow("Rectified + graph + stones", rect_graph)
            else:
                print("Detected", len(nodes), "nodes (expected 24).")

        cv2.imshow("Live", disp)
        cv2.imshow("Edges", edges)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()