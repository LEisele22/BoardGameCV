import cv2
import numpy as np

BOARD_SIZE = 600  # warped board will be 600x600



# Grid coordinates in a 0..6 x 0..6 grid
GRID_COORDS = [
    # outer ring
    (0,0),(3,0),(6,0),(6,3),(6,6),(3,6),(0,6),(0,3),
    # middle ring
    (1,1),(3,1),(5,1),(5,3),(5,5),(3,5),(1,5),(1,3),
    # inner ring
    (2,2),(3,2),(4,2),(4,3),(4,4),(3,4),(2,4),(2,3)
]

EDGES = [
    # outer ring cycle
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),
    # middle ring cycle
    (8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,8),
    # inner ring cycle
    (16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,16),
    # spokes outer -> middle
    (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15),
    # spokes middle -> inner
    (8,16),(9,17),(10,18),(11,19),(12,20),(13,21),(14,22),(15,23),
]


def canonical_nodes(board_size=BOARD_SIZE):
    """Return the 24 canonical node coordinates in warped board space."""
    step = board_size / 6.0   # 0..6 grid -> 0..board_size
    nodes = [(x*step, y*step) for (x, y) in GRID_COORDS]
    return np.array(nodes, dtype=np.float32)



def find_board_square(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 40, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    img_area = h * w

    best = None
    best_area = 0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        if area < 0.05 * img_area or area > 0.9 * img_area:
            # too small or basically the whole frame
            continue

        x, y, bw, bh = cv2.boundingRect(approx)
        if bh == 0:
            continue
        ratio = bw / float(bh)
        if ratio < 0.8 or ratio > 1.2:
            # not square enough
            continue

        if area > best_area:
            best_area = area
            best = approx

    return best


def order_corners(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


# -------------------------------------------------
# Main function: detect board, warp, overlay graph
# -------------------------------------------------
def process_board(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image:", image_path)
        return

    # Optional: downscale huge photos for speed
    max_side = 1200
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    board_cnt = find_board_square(img)
    if board_cnt is None:
        print("Could not find a square board contour.")
        cv2.imshow("Input", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    src = order_corners(board_cnt)
    dst = np.array([[0, 0],
                    [BOARD_SIZE, 0],
                    [BOARD_SIZE, BOARD_SIZE],
                    [0, BOARD_SIZE]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = np.linalg.inv(M)

    warped = cv2.warpPerspective(img, M, (BOARD_SIZE, BOARD_SIZE))

    # Canonical nodes in warped space
    nodes_warped = canonical_nodes()

    # Transform them back into original image space
    nodes_warped_h = nodes_warped.reshape(-1, 1, 2)
    nodes_orig_h = cv2.perspectiveTransform(nodes_warped_h, Minv)
    nodes_orig = nodes_orig_h.reshape(-1, 2)

    # Draw overlay on original image
    overlay = img.copy()

    # Draw edges
    for a, b in EDGES:
        x1, y1 = nodes_orig[a]
        x2, y2 = nodes_orig[b]
        cv2.line(overlay,
                 (int(x1), int(y1)),
                 (int(x2), int(y2)),
                 (0, 0, 255), 3)

    # Draw nodes
    for (x, y) in nodes_orig:
        cv2.circle(overlay, (int(x), int(y)), 7, (0, 255, 0), -1)

    # Also show warped board with graph in canonical space
    warped_overlay = warped.copy()
    for a, b in EDGES:
        x1, y1 = nodes_warped[a]
        x2, y2 = nodes_warped[b]
        cv2.line(warped_overlay,
                 (int(x1), int(y1)),
                 (int(x2), int(y2)),
                 (0, 0, 255), 3)
    for (x, y) in nodes_warped:
        cv2.circle(warped_overlay, (int(x), int(y)), 7, (0, 255, 0), -1)

    cv2.imshow("Original + Morris graph", overlay)
    cv2.imshow("Warped board + canonical graph", warped_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change this to your actual photo filename
    process_board("mills.webp")
