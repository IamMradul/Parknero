import cv2
import numpy as np

def analyze_spiral(image_path):
    """
    Analyzes a spiral drawing image to extract digital biomarkers related to handwriting.
    Returns a tuple: (dictionary of kinematic/visual features, processed_image_numpy_array)
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image. Ensure it's a valid format.")
        
    # Resize for consistency
    img = cv2.resize(img, (500, 500))
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Thresholding (Otsu's binarization)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (the drawn lines)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No drawing detected in the image.")
        
    # Assume the longest contour is the spiral
    main_contour = max(contours, key=cv2.contourArea)
    
    # Feature 1: Drawing length (perimeter)
    length = cv2.arcLength(main_contour, False)
    
    # Feature 2: Tremor/Smoothness (comparing arc length to a smoothed version)
    epsilon = 0.005 * length
    approx = cv2.approxPolyDP(main_contour, epsilon, False)
    smooth_length = cv2.arcLength(approx, False)
    tremor_index = (length / smooth_length) if smooth_length > 0 else 1.0
    
    # Feature 3: Pressure variability proxy (using Distance Transform to find line thickness)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    line_thicknesses = dist_transform[thresh > 0]
    
    if len(line_thicknesses) == 0:
        mean_thickness = 0
        thickness_variance = 0
    else:
        mean_thickness = np.mean(line_thicknesses)
        thickness_variance = np.var(line_thicknesses) 
        
    # Feature 4: Bounding box area
    x, y, w, h = cv2.boundingRect(main_contour)
    bounding_area = w * h
    
    # --- CREATE EXPLAINABLE VISUAL OVERLAY ---
    # Convert grayscale to color BGR so we can draw colored lines on it
    processed_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Draw the detected contour in bright green
    cv2.drawContours(processed_img, [main_contour], -1, (0, 255, 0), 2)
    # Draw a bounding box around it in blue
    cv2.rectangle(processed_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    feats = {
        "tremor_index": round(float(tremor_index), 4),
        "mean_thickness": round(float(mean_thickness), 4),
        "thickness_variance": round(float(thickness_variance), 4),
        "drawing_length": round(float(length), 4),
        "spatial_extent": float(bounding_area)
    }
    
    return feats, processed_img
