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

# Keep using the exact same edges list and 모든 methods – nothing changed.


def main_single_image(image_path):
    # Load image
    frame = cv2.imread(image_path)
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

    # --- Detect nodes ---
    nodes = detect_nodes(rect)
    if len(nodes) != EXPECT:
        print(f"Detected {len(nodes)} nodes, expected 24.")
        return len(nodes)

    # Sort nodes into fixed Morris ordering
    full_nodes = sort_nodes_morris_grid(nodes)

    # --- Draw graph for debugging ---
    rect_graph = rect.copy()
    rect_graph = draw_graph(rect_graph, full_nodes)

    # --- Detect stones ---
    stone_states = detect_all_stones(rect, full_nodes)

    # --- Overlay stones ---
    for state, (x, y) in zip(stone_states, full_nodes):
        if state == "black":
            cv2.circle(rect_graph, (int(x), int(y)), 10, (0, 0, 0), -1)
        elif state == "white":
            cv2.circle(rect_graph, (int(x), int(y)), 10, (255, 255, 255), -1)
        else:
            cv2.circle(rect_graph, (int(x), int(y)), 6, (0, 255, 0), 2)

    # Print result arrays
    w_pieces = [i for i, s in enumerate(stone_states) if s == "white"]
    b_pieces = [i for i, s in enumerate(stone_states) if s == "black"]
    print("White pieces:", w_pieces)
    print("Black pieces:", b_pieces)

    # Show debugging windows
    # cv2.imshow("Original", frame)
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Rectified", rect)
    # cv2.imshow("Rectified + Graph + Stones", rect_graph)

    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return len(nodes)


if __name__ == "__main__":
    acc = 0
    for i in range(26,51):
    # Change this path as needed
        nodes = main_single_image(f"metrics/images/IMG-20251207-WA00{i}.jpg")
        acc += nodes/24
    print(acc/25)
