"""
detect_corners.py
-----------------
Step 1: Automatic Registration Marker (Fiducial Marker) Detection
Detects the 4 outermost black square markers on the exam sheet and computes
their center points with sub-pixel precision using OpenCV moments.
"""

import cv2
import numpy as np
import os


def find_corner_markers(image_path: str, output_debug_path: str = None):
    """
    Detects the 4 outermost corner squares on an exam sheet image.
    
    Returns:
        src_corners (np.ndarray): 4x2 float array of [TL, TR, BR, BL] centers.
        annotated_image (np.ndarray): Image with markers visually highlighted.
    """
    # 1. Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"Loaded image: {image_path} (Resolution: {w}x{h})")

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Binarization (Otsu's Thresholding)
    # Otsu automatically finds the optimal threshold separating dark ink from bright paper.
    # THRESH_BINARY_INV makes dark ink pixels 255 (white) and background paper 0 (black).
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Find all external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found on page: {len(contours)}")

    # 5. Filter contours for solid square fiducial markers
    candidates = []
    min_area = 50                 # Ignore tiny dots (letters, commas, dust)
    max_area = (w * h) * 0.05     # Ignore huge borders

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        aspect_ratio = bw / float(bh)
        extent = area / float(bw * bh)

        # A printed square marker has:
        # - Aspect ratio close to 1.0 (0.80 <= w/h <= 1.25)
        # - High extent (solid ink fills >= 75% of bounding box)
        if 0.80 <= aspect_ratio <= 1.25 and extent >= 0.75:
            # Compute center of mass using spatial moments
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = bx + bw / 2.0
                cy = by + bh / 2.0

            candidates.append({
                "contour": c,
                "bbox": (bx, by, bw, bh),
                "center": (cx, cy),
                "area": area,
            })

    print(f"Square-like candidate markers found: {len(candidates)}")
    if len(candidates) < 4:
        raise ValueError(f"Could not find at least 4 corner markers! Found only {len(candidates)}.")

    # 6. Isolate the 4 outermost corners: TL, TR, BR, BL
    # Using coordinate projections:
    pts = np.array([c["center"] for c in candidates])

    # Sum (x + y): TL has min sum, BR has max sum
    s = pts[:, 0] + pts[:, 1]
    tl_idx = np.argmin(s)
    br_idx = np.argmax(s)

    # Difference (y - x): TR has min diff (large x, small y), BL has max diff (small x, large y)
    d = pts[:, 1] - pts[:, 0]
    tr_idx = np.argmin(d)
    bl_idx = np.argmax(d)

    corners = {
        "TL": candidates[tl_idx],
        "TR": candidates[tr_idx],
        "BR": candidates[br_idx],
        "BL": candidates[bl_idx]
    }

    src_corners = np.array([
        corners["TL"]["center"],
        corners["TR"]["center"],
        corners["BR"]["center"],
        corners["BL"]["center"]
    ], dtype=np.float32)

    # 7. Create visual debug image
    annotated = img.copy()
    colors = {
        "TL": (0, 0, 255),    # Red
        "TR": (0, 255, 0),    # Green
        "BR": (255, 0, 0),    # Blue
        "BL": (0, 255, 255)   # Yellow
    }

    for label, item in corners.items():
        bx, by, bw, bh = item["bbox"]
        cx, cy = item["center"]
        color = colors[label]

        # Draw green bounding box around marker
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        # Draw red center circle
        cv2.circle(annotated, (int(round(cx)), int(round(cy))), 5, color, -1)
        # Draw label text
        cv2.putText(
            annotated, f"{label} ({int(cx)},{int(cy)})",
            (bx - 10, by - 10 if by > 20 else by + bh + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
        )

    print("\n--- Detected 4 Outer Corner Markers ---")
    for label in ["TL", "TR", "BR", "BL"]:
        cx, cy = corners[label]["center"]
        bbox = corners[label]["bbox"]
        print(f"  {label}: Center = ({cx:.1f}, {cy:.1f}), Bounding Box = {bbox}")

    if output_debug_path:
        cv2.imwrite(output_debug_path, annotated)
        print(f"\nSaved debug visual to: {output_debug_path}")

    return src_corners, annotated


if __name__ == "__main__":
    sample_path = "./data/sample1.jpg"
    debug_output = "./data/sample1_corners_detected.jpg"
    src_corners, _ = find_corner_markers(sample_path, debug_output)
