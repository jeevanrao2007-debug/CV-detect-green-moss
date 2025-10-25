import cv2
import numpy as np
import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Detect green moss in rock images")
    p.add_argument("--image", "-i", required=True, help="Input rock image path")
    p.add_argument("--out", "-o", default="moss_result.jpg", help="Output annotated image path")
    p.add_argument("--mask", "-m", default="moss_mask.jpg", help="Binary mask output path")
    p.add_argument("--debug", action="store_true", help="Show debug windows")
    return p.parse_args()

def detect_green_moss(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = img.copy()
    cv2.drawContours(annotated, contours, -1, (0,255,0), 2)
    overlay = annotated.copy()
    overlay[mask > 0] = (0, 255, 0)
    annotated = cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0)
    return mask, annotated, len(contours)

def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        print("❌ ERROR: Could not load image:", args.image)
        sys.exit(1)
    mask, annotated, n_contours = detect_green_moss(img)
    cv2.imwrite(args.mask, mask)
    cv2.imwrite(args.out, annotated)
    print(f"✅ Saved mask -> {args.mask}")
    print(f"✅ Saved annotated -> {args.out}")
    print(f"🟢 Detected {n_contours} moss regions")
    if args.debug:
        cv2.imshow("Original", img)
        cv2.imshow("Moss Mask", mask)
        cv2.imshow("Annotated", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
