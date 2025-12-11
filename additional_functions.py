import math
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import cv2

import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile, DatasetWriter
from rasterstats import zonal_stats

from shapely.geometry import Polygon, MultiPolygon, box as box_geom
from shapely.geometry import base as shapely_geom
import shapely.affinity
from shapely.strtree import STRtree
from rtree import index

# ==============================
# 1. Basic Geometric Utilities
# =============================
def make_label_box(center: Tuple[float, float], width: float, height: float) -> Polygon:
    """
    Constructs a rectangular label geometry (bounding box) based on the center point 
    and given dimensions (width and height). The resulting rectangle is aligned 
    with the coordinate axes.

    Parameters:
    - center: Tuple (x, y) representing the center point of the label.
    - width: Float, width of the label box.
    - height: Float, height of the label box.

    Returns:
    - Polygon: Shapely Polygon object representing the rectangular label box.
    """

    # Extract the x and y coordinates of the center
    x, y = center

    # Construct a rectangle centered at (x, y) by calculating the four corner coordinates
    return Polygon([
        (x - width / 2, y - height / 2),  # bottom-left
        (x + width / 2, y - height / 2),  # bottom-right
        (x + width / 2, y + height / 2),  # top-right
        (x - width / 2, y + height / 2)   # top-left
    ])

def calculate_centroid(polygon: Union[Polygon, MultiPolygon]) -> Tuple[float, float]:
    """
    Calculates the centroid (center of mass) of a given polygon geometry 
    using the standard polygon centroid formula. In the case of a MultiPolygon,
    only the largest polygon (by area) is used for calculation.

    Parameters:
    - polygon: Shapely Polygon or MultiPolygon geometry.

    Returns:
    - Tuple (xc, yc): Coordinates of the centroid.
    """

    # If the geometry is a MultiPolygon, use the largest constituent polygon
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda p: p.area)

    # Check for correct input type
    if not isinstance(polygon, Polygon):
        raise ValueError("Geometry must be a Polygon or MultiPolygon")

    # Extract exterior boundary coordinates
    coords = polygon.exterior.coords
    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])

    # Ensure the polygon is explicitly closed
    if not (x[0] == x[-1] and y[0] == y[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Compute signed area of the polygon using the shoelace formula
    A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    # Compute centroid coordinates (weighted average of vertex coordinates)
    xc = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    yc = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))

    return xc, yc

def calculate_angle_and_distance(vertex_a: Tuple[float, float], vertex_b: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculates the angle and Euclidean distance between two points.

    Parameters:
    - vertex_a: tuple of floats (x, y). Coordinates of the first point.
    - vertex_b: tuple of floats (x, y). Coordinates of the second point.

    Returns:
    - angle: float. Angle in degrees between the two points, measured counterclockwise from the x-axis.
    - distance: float. Euclidean distance between the two points.
    """
    # Difference in coordinates
    dx = vertex_a[0] - vertex_b[0]
    dy = vertex_a[1] - vertex_b[1]

    # Compute angle in degrees using atan2
    angle = math.degrees(math.atan2(dy, dx))

    # Compute Euclidean distance
    distance = math.sqrt(dx**2 + dy**2)

    return angle, distance

def angular_distance(a: float, b: float) -> float:
    """
    Computes the shortest angular distance between two angles on a circle.

    Parameters:
    - a: float. First angle in degrees.
    - b: float. Second angle in degrees.

    Returns:
    - float: Smallest angular difference between the two angles, in degrees.
    """

    # Return the minimum of the direct and wrap-around distances
    return min(abs(a - b), 360 - abs(a - b))

def compute_label_dimensions(font_size_pt: float, map_scale: float, n_chars: int):
    """
    Function to compute the height and width of a map label in meters based on 
    font size and map scale.

    Parameters:
    - font_size_pt: Font size F in points.
    - map_scale: Map scale denominator M (e.g., 10000 for a 1:10,000 map).
    - n_chars: Number of characters n in the label text.

    Returns:
    - (h, w): Tuple containing the label height (h) and width (w) in meters.
    """
    POINTS_PER_INCH = 72.0         # Number of points per inch
    INCH_TO_METER = 0.0254         # Conversion factor from inches to meters
    AVG_CHAR_WIDTH_FACTOR = 0.6    # Approx. character width relative to height

    # Height of the label in meters:
    # h = (F / 72) * 0.0254 / M
    h = (font_size_pt / POINTS_PER_INCH) * INCH_TO_METER * map_scale

    # Width of the label in meters:
    # w = n * (0.6 * F/72 * 0.0254 / M)
    w = n_chars * (AVG_CHAR_WIDTH_FACTOR * font_size_pt / POINTS_PER_INCH * INCH_TO_METER * map_scale)

    return int(h), int(w)
# ==============================
# 2. Direction Classification
# ==============================
    
def classify_direction_by_angle(angle: float) -> Tuple[str, str]:
    """
    Classifies a given angle into the two most relevant cartographic directions
    (e.g., right, up, left, down) based on angular proximity to quadrant sectors.

    Parameters:
    - angle: float. The directional angle in degrees (e.g., between a label and its corresponding object).

    Returns:
    - List of two direction strings (e.g., ['right', 'up']) representing the closest
      directional sectors to the input angle, sorted by proximity.
    """

    # Normalize the angle to the [0, 360) range
    angle = (angle + 360) % 360

    # Define angular ranges for each cardinal direction
    ranges = {
        'right': (-45, 45),
        'up':    (45, 135),
        'left':  (135, -135),
        'down':  (-135, -45),
    }

    direction_distances = []

    # For each direction, calculate the angular midpoint and compute distance to the input angle
    for direction, (start, end) in ranges.items():
        start = (start + 360) % 360
        end = (end + 360) % 360

        # Handle angle wrap-around correctly
        if start < end:
            center = (start + end) / 2
        else:
            center = ((start + end + 360) / 2) % 360

        dist = angular_distance(angle, center)
        direction_distances.append((direction, dist))

    # Return the top two closest directions by angular proximity
    direction_distances.sort(key=lambda x: x[1])

    return (direction_distances[0][0], direction_distances[1][0])

def classify_direction_priority(angle_deg: float) -> Tuple[str, float]:
    """
    Classifies a given angle (in degrees) into one of eight directional sectors
    and assigns a priority weight based on empirical frequency analysis.

    Parameters:
    - angle_deg: float. Directional angle in degrees, typically derived from the
      line between the centroid of a label and the centroid of its target feature.

    Returns:
    - Tuple (str, float): Direction label (e.g., 'right') and its associated
      statistical priority based on frequency in the ZTM25 dataset.
    """

    # Define angular boundaries for 8 directional sectors
    direction_sectors = [
        ("right",        -22.5,   22.5),
        ("top-right",     22.5,   67.5),
        ("top",           67.5,  112.5),
        ("top-left",     112.5,  157.5),
        ("left",         157.5,  180.0),
        ("left",        -180.0, -157.5),
        ("bottom-left", -157.5, -112.5),
        ("bottom",      -112.5,  -67.5),
        ("bottom-right", -67.5,  -22.5),
    ]

    # Frequency-derived priority scores based on analysis of >6500 labels from ZTM25
    direction_priority = {
        "right":        1596 / 6518,
        "top-right":    1182 / 6518,
        "top":           779 / 6518,
        "top-left":      686 / 6518,
        "left":          496 / 6518,
        "bottom-left":   454 / 6518,
        "bottom":        451 / 6518,
        "bottom-right":  874 / 6518
    }

    # Match the input angle to the appropriate direction sector
    for name, start, end in direction_sectors:
        if start <= angle_deg < end:
            return name, direction_priority[name]

# ==============================
# 3. Label Box Manipulation
# ==============================
def push_label(box: Polygon, polygon: Polygon, direction: str, offset_distance: float) -> Union[Polygon, None]:
    """
    Shifts a label box in a given direction relative to a polygon, ensuring it aligns with
    the outline and maintains a specific offset distance.

    Parameters:
    - box: Polygon representing the label to be moved.
    - polygon: Polygon to which the label should be associated (typically a settlement).
    - direction: String, one of ["left", "right", "up", "down"].
    - offset_distance: Float, required spacing between the polygon edge and the label box.

    Returns:
    - Translated Polygon if placement is possible; otherwise None.
    """
    minx, miny, maxx, maxy = box.bounds
    coords = list(polygon.exterior.coords)
    candidate_values = []

    # Horizontal shift (left/right)
    if direction in ["left", "right"]:
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            # Skip segments outside the vertical extent of the label
            if max(y1, y2) < miny or min(y1, y2) > maxy:
                continue

            points = []

            # Retain segment endpoints if they lie within vertical bounds
            if miny <= y1 <= maxy:
                points.append((x1, y1))
            if miny <= y2 <= maxy:
                points.append((x2, y2))

            # Intersect segment with top and bottom edges of the label
            if (min(y1, y2) <= miny <= max(y1, y2)) and y1 != y2:
                t = (miny - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                points.append((x, miny))
            if (min(y1, y2) <= maxy <= max(y1, y2)) and y1 != y2:
                t = (maxy - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                points.append((x, maxy))

            # Select optimal x-coordinates for offset computation
            if points:
                x_values = [pt[0] for pt in points]
                if direction == "left":
                    best_x = max(x_values)  # Closest left-edge of polygon
                else:
                    best_x = min(x_values)  # Closest right-edge of polygon
                candidate_values.append(best_x)

        # If any candidate positions found, compute shift
        if candidate_values:
            if direction == "right":
                outline_x = min(candidate_values)
                target_left_x = outline_x - offset_distance
                shift = target_left_x - maxx
            else:
                outline_x = max(candidate_values)
                target_left_x = outline_x + offset_distance
                shift = target_left_x - minx

            # Apply horizontal translation
            shifted_box = shapely.affinity.translate(box, xoff=shift, yoff=0)
            return shifted_box

    # Vertical shift (up/down)
    if direction in ["up", "down"]:
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            # Skip segments outside horizontal extent of the label
            if max(x1, x2) < minx or min(x1, x2) > maxx:
                continue

            points = []

            # Retain segment endpoints if they lie within horizontal bounds
            if minx <= x1 <= maxx:
                points.append((x1, y1))
            if minx <= x2 <= maxx:
                points.append((x2, y2))

            # Intersect segment with left and right edges of the label
            if (min(x1, x2) <= minx <= max(x1, x2)) and x1 != x2:
                t = (minx - x1) / (x2 - x1)
                y = y1 + t * (y2 - y1)
                points.append((minx, y))
            if (min(x1, x2) <= maxx <= max(x1, x2)) and x1 != x2:
                t = (maxx - x1) / (x2 - x1)
                y = y1 + t * (y2 - y1)
                points.append((maxx, y))

            # Select optimal y-coordinates for offset computation
            if points:
                y_values = [pt[1] for pt in points]
                if direction == "down":
                    best_y = max(y_values)
                else:
                    best_y = min(y_values)
                candidate_values.append(best_y)

        # If any candidate positions found, compute shift
        if candidate_values:
            if direction == "up":
                outline_y = min(candidate_values)
                target_bottom_y = outline_y - offset_distance
                shift = target_bottom_y - maxy
            else:
                outline_y = max(candidate_values)
                target_bottom_y = outline_y + offset_distance
                shift = target_bottom_y - miny

            # Apply vertical translation
            shifted_box = shapely.affinity.translate(box, xoff=0, yoff=shift)
            return shifted_box

    # No valid shift found
    return None


# ==============================
# 4. Sweep-Line Candidate Generation
# ==============================
def find_segment_line_pairs(line_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]], d: float, direction: str
                            ) -> List[Tuple[Tuple[Tuple[float, float], Tuple[float, float]], float]]:
    """
    Finds pairs of line segments and sweep-line positions where they potentially intersect.

    This function simulates a generalized sweep-line algorithm, processing a set of input 
    line segments and generating positions at regular intervals (defined by `d`) 
    where candidate label placements will be evaluated.

    Parameters:
    - line_segments: List of tuples of endpoints ((x1, y1), (x2, y2)), representing line segments.
    - d: float. Step distance for the sweep line (e.g., every 10 meters).
    - direction: str. Either 'vertical' or 'horizontal', indicating the direction of sweep.

    Returns:
    - List of tuples: Each tuple contains a line segment and the sweep-line coordinate (x or y)
                      where the segment intersects the sweep line.
    """
    # Determine orientation-specific settings
    if direction == "vertical":
        get_coord = lambda pt: pt[1]
        event_type_upper = "upper"
        event_type_lower = "lower"
        # Sweep top-down
        sort_reverse = True
    elif direction == "horizontal":
        get_coord = lambda pt: pt[0]
        event_type_upper = "left"
        event_type_lower = "right"
        # Sweep left-right
        sort_reverse = False
    else:
        raise ValueError("Direction must be 'vertical' or 'horizontal'.")

    # Step 1: Create event points for sweep-line simulation
    event_points = set()
    for seg in line_segments:
        event_points.add((get_coord(seg[0]), event_type_upper, seg))
        event_points.add((get_coord(seg[1]), event_type_lower, seg))

    # Sort event queue according to sweep direction
    event_points = sorted(event_points, reverse=sort_reverse)
    # Active segment list
    R = []  
    pairs = []

    # Start from the first event's coordinate
    prev_coord = event_points[0][0]

    # Step 2: Sweep through events
    for curr_coord, kind, seg in event_points:
        if kind == event_type_upper:
            R.append(seg)
            # Sort active segments for deterministic output
            R.sort(key=lambda s: s[0][0 if direction == "vertical" else 1])

        elif kind == event_type_lower:
            # If we moved the sweep line since the last event
            if prev_coord != curr_coord:
                n = int(np.floor(abs(prev_coord - curr_coord) / d))
                if n != 0:
                    # Avoid duplicating sweep line on first iteration
                    j = 1 if prev_coord == event_points[0][0] else 0
                    for i in range(j, n + 1):
                        pos = prev_coord - d * i if direction == "vertical" else prev_coord + d * i
                        for t in R:
                            c1 = get_coord(t[0])
                            c2 = get_coord(t[1])
                            if (c1 >= pos >= c2) or (c2 >= pos >= c1):
                                pairs.append((t, pos))
                    prev_coord = pos
            if seg in R:
                R.remove(seg)

    last_seg = event_points[0][2]
    last_coord = get_coord(last_seg[0])
    pairs.append((last_seg, last_coord))

    # Final sorting to maintain consistency
    pairs.sort(key=lambda pair: pair[1], reverse=(direction == "vertical"))
    return pairs

def find_candidate_positions(segment_line_pairs: List[Tuple[Tuple[float, float], float]], label_width: float, label_height: float, direction: str) -> List[Polygon]:
    """
    Computes candidate label positions at the intersection of polygon edges and sweeping lines.

    This function is used in a sweep-line algorithm to generate candidate label positions
    by calculating intersection points between polygon edges and either vertical or horizontal lines,
    and placing rectangular label boxes centered at these intersections.

    Parameters:
    - segment_line_pairs: List of tuples ((p1, p2), coord), where:
        - p1, p2 are the endpoints of a line segment (edge of polygon).
        - coord is the fixed x or y value of the sweep line intersecting that segment.
    - label_width: Width of the candidate label box in map units.
    - label_height: Height of the candidate label box in map units.
    - direction: Sweep direction – either 'vertical' (horizontal lines) or 'horizontal' (vertical lines).

    Returns:
    - List[Polygon]: A list of Shapely rectangular polygons representing candidate label boxes,
      each centered at a segment–sweep line intersection.
    """
    candidate_positions = []

    for seg, coord in segment_line_pairs if direction == "horizontal" else segment_line_pairs[:-1]:
        p1, p2 = seg
        x1, y1 = p1
        x2, y2 = p2

        if direction == "vertical":
            # Solve for x given a constant y = coord (horizontal sweep line)
            if x1 == x2:
                # Vertical segment — intersection is direct
                x_intersect = x1
            else:
                # Compute intersection x using y = mx + b => x = (coord - b) / m
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                x_intersect = (coord - b) / m
            center = (x_intersect, coord)

        elif direction == "horizontal":
            # Solve for y given a constant x = coord (vertical sweep line)
            if y1 == y2:
                # Horizontal segment — intersection is direct
                y_intersect = y1
            else:
                # Compute intersection y using y = mx + b
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                y_intersect = m * coord + b
            center = (coord, y_intersect)

        else:
            raise ValueError("Direction must be 'vertical' or 'horizontal'.")

        # Construct a label box centered at the computed intersection point
        candidate_box = make_label_box(center, label_width, label_height)
        candidate_positions.append(candidate_box)

    return candidate_positions

def filter_additional_candidates(
    candidates: List[Polygon],
    existing_boxes: List[Polygon],
    threshold: float
) -> List[Polygon]:
    """
    Filters label boxes using axis-aligned bounding box overlap checks (no spatial index).

    Parameters:
    - candidates: New candidate label boxes (as Polygons).
    - existing_boxes: Already accepted label boxes (as Polygons).
    - threshold: Max allowed overlap ratio (0–1) for rejection.

    Returns:
    - List of accepted boxes including previous and new.
    """
    for candidate in candidates:
        c_minx, c_miny, c_maxx, c_maxy = candidate.bounds
        candidate_area = candidate.area

        overlaps = False

        for existing in existing_boxes:
            e_minx, e_miny, e_maxx, e_maxy = existing.bounds

            # Zkontroluj, zda se překrývají na obou osách
            x_overlap = not (c_maxx <= e_minx or c_minx >= e_maxx)
            y_overlap = not (c_maxy <= e_miny or c_miny >= e_maxy)

            if x_overlap and y_overlap:
                intersection = candidate.intersection(existing)
                if intersection.area / candidate_area > threshold:
                    overlaps = True
                    break

        if not overlaps:
            existing_boxes.append(candidate)

    return existing_boxes

def generate_label_candidates(gdf: gpd.GeoDataFrame, M: int, font_size: Optional[float] = None) -> gpd.GeoDataFrame:
    """
    Generates candidate label placements around polygon geometries using a sweep-line approach.

    This function buffers each input geometry and places rectangular label boxes along the
    perimeter using vertical and horizontal sweeps. The final candidates are filtered to
    avoid excessive overlaps.

    Parameters:
    - gdf (GeoDataFrame): Input geometries with associated label columns ('text', 'fontsize').
    - M (int): Map scale denominator (e.g., 25000 for 1:25,000).
    - font_size (int, optional): Fixed font size for all labels. If None, uses 'fontsize' column.

    Returns:
    - GeoDataFrame: Candidate label positions with reference to their source geometry.
    """
    if "fontsize" not in gdf.columns and font_size is None:
        raise ValueError("Input GeoDataFrame must contain 'fontsize' column if fontsize parameter is not provided.")

    BUFFER_MULTIPLIER = 2       # Controls buffer distance around each geometry
    OVERLAP_THRESHOLD = 0.4     # Max allowed overlap ratio between label boxes

    all_source_ids = []
    all_label_boxes = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        label_text = row.text

        if font_size is None:
            font_size = row.fontsize

        # Skip geometries without defined label size
        if np.isnan(font_size) or label_text is None or label_text == "":
            continue
        
        label_height, label_width = compute_label_dimensions(font_size, M, len(label_text))

        buffer_distance = max(label_width, label_height) * BUFFER_MULTIPLIER
        buffered_geom = geom.buffer(buffer_distance)

        # Extract boundary line segments from the buffered geometry
        segments = [
            (buffered_geom.exterior.coords[i], buffered_geom.exterior.coords[i + 1])
            for i in range(len(buffered_geom.exterior.coords) - 1)
        ]

        # Sort segments based on direction for sweep processing
        v_segments = [(p1, p2) if p1[1] >= p2[1] else (p2, p1) for (p1, p2) in segments]
        h_segments = [(p1, p2) if p1[0] <= p2[0] else (p2, p1) for (p1, p2) in segments]

        # Perform vertical sweep
        vertical_pairs = find_segment_line_pairs(v_segments, label_height, "vertical")
        vertical_candidates = find_candidate_positions(vertical_pairs, label_width, label_height, "vertical")
        
        # Perform horizontal sweep
        horizontal_pairs = find_segment_line_pairs(h_segments, label_width / 2, "horizontal")
        horizontal_candidates = find_candidate_positions(horizontal_pairs, label_width, label_height, "horizontal")

        # Filter horizontal candidates to avoid overlapping too much with vertical ones
        final_candidates = filter_additional_candidates(
            horizontal_candidates,
            vertical_candidates,
            threshold=OVERLAP_THRESHOLD
        )

        # Accumulate results
        all_label_boxes += final_candidates
        all_source_ids += [idx] * len(final_candidates)

    # Assemble results into a GeoDataFrame
    gdf_candidates = gpd.GeoDataFrame({
        "source_id": all_source_ids,
        "geometry": all_label_boxes
    }, crs=gdf.crs)

    return gdf_candidates
# ==============================
# 5. Raster Processing & Visual Metrics
# ==============================
def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """
    Calculates the relative luminance of an RGB image based on WCAG standards.

    Parameters:
    - rgb: np.ndarray of shape (H, W, 3). RGB image array with values in the range [0, 255].

    Returns:
    - np.ndarray of shape (H, W). Grayscale array representing the perceived brightness 
      (luminance) of each pixel, scaled between 0 and 1.
    """

    # Normalize RGB values to [0, 1] range
    rgb = rgb / 255.0

    # Split the normalized image into individual color channels
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Define adjustment function per channel as specified in WCAG:
    # Linearizes the sRGB components using a piecewise gamma correction
    def adjust(c):
        # Apply linear transformation for darker values (low gamma)
        # and power-law transformation for brighter ones
        return np.where(
            c <= 0.03928, 
            c / 12.92, 
            ((c + 0.055) / 1.055) ** 2.4
        )

    # Apply the adjustment to each color channel
    r, g, b = adjust(r), adjust(g), adjust(b)

    # Compute relative luminance using weighted sum of adjusted channels
    # These weights reflect human perception (green is most prominent)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def get_contrast_and_map_load(input_tif_path: str, text_color_rgb: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[DatasetWriter, DatasetWriter]:
    """
    Computes two visual metrics from a raster map image:
    1. Map load: based on edge density and general visual complexity.
    2. Contrast map: relative contrast between text color and background.

    Parameters:
    - input_tif_path: str. Path to the RGB raster file (GeoTIFF format).
    - text_color_rgb: list of three integers. RGB values of the label text color (default: black).

    Returns:
    - grayscale_ds: In-memory raster dataset representing map load.
    - contrast_ds: In-memory raster dataset representing relative contrast for each pixel.
    """
    # Open RGB raster file
    with rasterio.open(input_tif_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype=rasterio.uint8)

        # Read RGB channels and stack them
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)
        rgb_image = np.stack((red, green, blue), axis=-1)

    # Convert RGB to grayscale using ITU-R BT.601 standard
    grayscale = (0.299 * red + 0.587 * green + 0.114 * blue).astype(np.uint8)

    # Apply Sobel filters to detect edges in X and Y directions
    sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sobelx**2 + sobely**2)
    edge = cv2.convertScaleAbs(edge)

    # Smooth edges using a 3x3 Gaussian blur kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0
    filtered = cv2.filter2D(edge, -1, kernel)

    # Normalize edge intensity to a scale of 0–100 for interpretability
    normalized = (filtered.astype(np.float32) / 255.0) * 100

    # Compute relative luminance of the background using the same RGB standard
    bg_luminance = relative_luminance(rgb_image)

    # Normalize input text color and compute its luminance
    text_rgb = np.array(text_color_rgb) / 255.0
    text_lum = relative_luminance(text_rgb[np.newaxis, np.newaxis, :])[0, 0]

    # Compute relative contrast according to WCAG definition
    L1 = np.maximum(bg_luminance, text_lum)
    L2 = np.minimum(bg_luminance, text_lum)
    contrast_map = ((L1 + 0.05) / (L2 + 0.05)).astype(np.float32)

    # Save normalized grayscale (map load) to in-memory raster
    profile.update(dtype=rasterio.uint8)
    grayscale_mem = MemoryFile()
    grayscale_ds = grayscale_mem.open(**profile)
    grayscale_ds.write(normalized, 1)

    # Save contrast map to in-memory raster
    profile.update(dtype=rasterio.float32)
    contrast_mem = MemoryFile()
    contrast_ds = contrast_mem.open(**profile)
    contrast_ds.write(contrast_map, 1)

    return grayscale_ds, contrast_ds

# ==============================
# 6. Final Label Placement
# ==============================
def get_final_label_positions(gdf_sidla: gpd.GeoDataFrame, gdf_labels: gpd.GeoDataFrame, offset_distance: float, incrementation: float, 
                              collision: Union[gpd.GeoDataFrame, shapely_geom.BaseGeometry] = None) -> gpd.GeoDataFrame:
    """
    Function to finalize label placement by pushing candidate boxes toward the feature centroid
    while avoiding overlaps with other features, collisions, or labels.

    Parameters:
    - gdf_sidla (GeoDataFrame): Polygon features representing settlement areas.
    - gdf_labels (GeoDataFrame): Candidate label positions associated by 'source_id'.
    - offset_distance (float): Initial distance to push labels from outline inward.
    - incrementation (float): Incremental fallback distance if no valid labels are found.
    - collision (GeoDataFrame or Shapely geometry, optional): Obstacles to avoid (e.g., roads, rivers).

    Returns:
    - final_labels (GeoDataFrame): All successfully positioned labels with scores and metadata.
    """
    # Allow up to 65% overlap with other candidate labels
    INTERSECTION_THRESHOLD = 0.65

    all_adjusted_labels = []    

    # Prepare the geometry pool to avoid overlaps with
    if collision is not None:
        if isinstance(collision, gpd.GeoDataFrame):
            collision_geoms = list(collision.geometry.values)
        elif isinstance(collision, shapely_geom.BaseGeometry):
            collision_geoms = [collision]
        else:
            raise TypeError("Unsupported type for 'collision': expected GeoDataFrame or Shapely geometry.")

        combined_geometries = list(gdf_sidla.geometry.values) + collision_geoms
    else:
        combined_geometries = list(gdf_sidla.geometry.values)

    # Spatial index for collision detection
    exclusion_tree = STRtree(combined_geometries)

    # Iterate over all settlements (one at a time)
    for idx, settlement in gdf_sidla.iterrows():
        polygon = settlement.geometry
        centroid = calculate_centroid(polygon)

        # Candidate positions of current settlement polygon
        matching_labels = gdf_labels[gdf_labels["source_id"] == idx]

        # Valid label boxes placed for this settlement
        adjusted_geoms = []
        direction_scores = []
        distances_to_centroid = []

        # Build the spatial index
        tree = index.Index()
        i = 0

        for _, row in matching_labels.iterrows():
            box = row.geometry
            angle, _ = calculate_angle_and_distance(centroid, (box.centroid.x, box.centroid.y))

            # Determine top 2 directions toward centroid
            directions = classify_direction_by_angle(angle)

            for direction in directions:
                new_box = push_label(box, polygon, direction, offset_distance)  # Attempt to shift toward polygon

                if new_box is None:
                    continue

                # Exclude boxes intersecting obstacles or polygon
                matches = exclusion_tree.query(new_box)
                if any(new_box.intersects(combined_geometries[i]) for i in matches):
                    continue

                # Prevent high-overlap with other already-placed boxes
                candidates = list(tree.intersection(new_box.bounds))
                overlap = any(
                    not new_box.intersection(adjusted_geoms[i]).is_empty and
                    new_box.intersection(adjusted_geoms[i]).area > INTERSECTION_THRESHOLD * new_box.area
                    for i in candidates
                )
                if overlap:
                    continue

                # Box is accepted, update tracking lists
                adjusted_geoms.append(new_box)

                # Rebuild index after each insertion
                tree.insert(i, new_box.bounds)
                i +=1

                angle, distance = calculate_angle_and_distance((new_box.centroid.x, new_box.centroid.y), centroid)
                _, priority = classify_direction_priority(angle)

                direction_scores.append(priority)
                distances_to_centroid.append(distance)

        # Append per-settlement results to global list
        adjusted_gdf = gpd.GeoDataFrame(
            {
                "source_id": idx,
                "dir_prio": direction_scores,
                "d_prio": direction_scores,
                "dist": distances_to_centroid,
                "geometry": adjusted_geoms
            },
            crs=gdf_labels.crs
        )
        all_adjusted_labels.append(adjusted_gdf)

    # Combine all settlements into one GeoDataFrame
    final_labels = gpd.GeoDataFrame(pd.concat(all_adjusted_labels, ignore_index=True), crs=gdf_labels.crs)
    final_labels.to_file("C:/diplomka/data/input data/benesovsko/rtree.shp")

    # If no valid labels were found, increase offset and retry
    if final_labels.empty:
        offset_distance += incrementation
        return get_final_label_positions(gdf_sidla, gdf_labels, offset_distance, incrementation, collision)
    else:
        return final_labels

# ==============================
# 7. Scoring and Optimization
# ==============================
def normalize(group: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes relevant metrics in a group of label candidates for a single source_id.

    This function ensures all scoring metrics are on a common [0,1] scale, allowing them
    to be combined via a weighted sum. For some metrics, lower values are better, and 
    these are inverted accordingly during normalization.

    Parameters:
    - group: Pandas DataFrame representing label candidates for one unique source_id.

    Returns:
    - DataFrame with normalized values (either updated in-place or added as new columns).
    """
    # Avoid modifying the original DataFrame
    group = group.copy()

    for column in ["contrast", "map_load", "dist", "dir_prio"]:

        if column == "dir_prio":
            min_val = 451/6518
            max_val = 1596/6518
        else:
            min_val = group[column].min()
            max_val = group[column].max()
        group[f"{column}_min"] = min_val
        group[f"{column}_max"] = max_val

        if min_val == max_val:
            # If all values are identical, assign 0 (neutral effect)
            group[column if column != "dist" else "dist_norm"] = 0.0
        else:
            # Normalize to [0,1] using min-max scaling
            norm_values = (group[column] - min_val) / (max_val - min_val)

            # Invert metrics where a lower value is better (e.g., contrast, dir_prio)
            if column in ["contrast", "dir_prio"]:
                norm_values = 1.0 - norm_values

            # Distance gets its own normalized column
            
            group[column + "_norm"] = norm_values

    return group

def resolve_overlaps(candidates: gpd.GeoDataFrame, group_col: str = "source_id", score_col: str = "score", geom_col: str = "geometry") -> gpd.GeoDataFrame:
    """
    Resolves overlapping label candidates by selecting the best-scoring one per group 
    and ensuring no two accepted labels overlap spatially.

    Parameters:
    - candidates: GeoDataFrame containing multiple label candidates with assigned scores.
    - group_col: str, column name identifying unique feature groups (e.g., settlements).
    - score_col: str, column name representing the evaluation score (lower is better).
    - geom_col: str, column name containing geometric information.

    Returns:
    - GeoDataFrame with non-overlapping label candidates, one per group.
    """
    
    resolved = []          # Final list of accepted (non-overlapping) label records
    resolved_geoms = []    # Geometries of already accepted labels (for spatial checks)

    # Group candidates by source_id and sort them by score (ascending)
    grouped = candidates.sort_values(score_col).groupby(group_col)

    # Convert each group into a list of namedtuples for efficient access
    remaining = {src: list(grp.itertuples(index=False)) for src, grp in grouped}

    # Iteratively resolve overlaps until all groups are either accepted or exhausted
    while any(remaining.values()):
        for src in list(remaining.keys()):
            if not remaining[src]:
                del remaining[src]
                continue

            # Pick the best-scoring candidate for the current group
            candidate = remaining[src].pop(0)
            geom = getattr(candidate, geom_col)

            # Check for spatial overlap with already resolved labels
            overlap_found = any(not geom.disjoint(existing) for existing in resolved_geoms)
            if overlap_found:
                continue  # Skip this candidate and check the next one in the group

            # Accept this candidate and store its geometry for future collision checks
            resolved.append(candidate)
            resolved_geoms.append(geom)

            # Group resolved, no need to check further candidates for this source_id
            del remaining[src]

    # Convert final list of namedtuples back into a GeoDataFrame
    return gpd.GeoDataFrame([r._asdict() for r in resolved], crs=candidates.crs)

def find_the_best_positions(contrast_ds: DatasetWriter, map_load_ds: DatasetWriter, final_labels: gpd.GeoDataFrame, 
                            w_con: int = 0.2, w_map: int = 0.4, w_dir: int = 0.1, w_dis: int = 0.3) -> gpd.GeoDataFrame:
    """
    Evaluates candidate label positions based on contrast, map load, direction priority, and distance.

    Parameters:
    - contrast_ds: Raster dataset of contrast values.
    - map_load_ds: Raster dataset of map load (visual density).
    - final_labels: GeoDataFrame with label geometry and attributes.
    - w_con, w_map, w_dir, w_dis: Weights for contrast, map load, direction priority, and distance.

    Returns:
    - GeoDataFrame with best positions resolved using total weighted score.
    """

    # Clip candidates to raster extent
    bounds = contrast_ds.bounds
    raster_bbox = box_geom(*bounds)
    final_labels = final_labels[final_labels.geometry.within(raster_bbox)].copy()

    # Extract raster arrays and transforms
    contrast_array = contrast_ds.read(1)
    contrast_transform = contrast_ds.transform
    contrast_nodata = contrast_ds.nodata if contrast_ds.nodata is not None else -999

    map_array = map_load_ds.read(1)
    map_transform = map_load_ds.transform
    map_nodata = map_load_ds.nodata if map_load_ds.nodata is not None else -999

    # Compute zonal statistics
    final_labels["contrast"] = [
        d["mean"] for d in zonal_stats(final_labels, contrast_array, affine=contrast_transform, nodata=contrast_nodata, stats=["mean"])
    ]
    final_labels["map_load"] = [
        d["mean"] for d in zonal_stats(final_labels, map_array, affine=map_transform, nodata=map_nodata, stats=["mean"])
    ]

    # Normalize values within each group
    final_labels = final_labels.groupby("source_id", group_keys=False).apply(normalize)
    final_labels.dropna(subset=["contrast", "map_load"], inplace=True)

    # Weighted scoring
    final_labels["score"] = (
        w_map * final_labels["map_load_norm"] +
        w_dir * final_labels["dir_prio_norm"] +
        w_dis * final_labels["dist_norm"] +
        w_con * final_labels["contrast_norm"]
    )
    #final_labels.to_file("C:/diplomka/data/input data/positions.shp")

    return resolve_overlaps(final_labels)
