import numpy as np
import cv2
from shapely.geometry import Polygon, LinearRing
from typing import Dict, List, Tuple
import pandas as pd

def find_nearest_vertices(exterior: LinearRing, hole: LinearRing) -> Tuple[Tuple[float, float], Tuple[float, float], int, int]:
    """
    Find the closest pair of vertices between exterior and hole rings.
    Returns the vertices coordinates and their indices.
    """
    ext_coords = list(exterior.coords)
    hole_coords = list(hole.coords)
    
    min_dist = float('inf')
    nearest_ext = None
    nearest_hole = None
    ext_idx = hole_idx = 0
    
    for i, ext_point in enumerate(ext_coords):
        for j, hole_point in enumerate(hole_coords):
            dist = np.sqrt((ext_point[0] - hole_point[0])**2 + 
                         (ext_point[1] - hole_point[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_ext = ext_point
                nearest_hole = hole_point
                ext_idx = i
                hole_idx = j
                
    return nearest_ext, nearest_hole, ext_idx, hole_idx

def get_ordered_vertices(ring: LinearRing, start_idx: int, is_ccw: bool) -> List[Tuple[float, float]]:
    """
    Get vertices from a ring starting at given index in specified direction.
    """
    coords = list(ring.coords)[:-1]  # Remove the last point which is same as first
    n = len(coords)
    
    if is_ccw:
        return coords[start_idx:] + coords[:start_idx]
    else:
        reversed_coords = coords[start_idx::-1] + coords[:start_idx:-1]
        return reversed_coords
    
def cut_polygon_recursive(polygon: Polygon) -> Polygon:
    """
    Recursively cut a polygon with holes into a simple polygon without holes.
    """
    # Base case: if polygon has no holes, return it
    if len(polygon.interiors) == 0:
        return polygon

    # Take the first hole and keep others for later
    first_hole = polygon.interiors[0]
    other_holes = list(polygon.interiors[1:])
    
    # Find nearest vertices between exterior and the hole
    nearest_ext, nearest_hole, ext_idx, hole_idx = find_nearest_vertices(
        polygon.exterior, first_hole
    )
    
    # Check if hole is clockwise or counterclockwise
    hole_is_ccw = LinearRing(first_hole).is_ccw

    # Build new exterior coordinates:
    # 1. Start from the nearest exterior vertex and go around
    exterior_vertices = get_ordered_vertices(polygon.exterior, ext_idx, True)
    
    # 2. Add the nearest exterior vertex again
    new_coords = exterior_vertices + [nearest_ext]
    
    # 3. Add hole vertices starting from nearest point in appropriate direction
    hole_vertices = get_ordered_vertices(first_hole, hole_idx, not hole_is_ccw)
    new_coords.extend(hole_vertices)
    
    # 4. Close the ring by adding the nearest vertices again
    new_coords.extend([nearest_hole, nearest_ext])

    # Create new polygon with remaining holes
    new_polygon = Polygon(new_coords, [list(hole.coords) for hole in other_holes])
    # Recursively process remaining holes
    return cut_polygon_recursive(new_polygon)


def mask_to_polygons(label_mask: np.ndarray, 
                     background_label = 0,
                    simple_cut: bool = False) -> Dict[int, List[Polygon]]:
    """
    Convert a label mask to a dictionary of polygonal contours for each unique label.
    
    Args:
        label_mask: 2D numpy array with uint8 labels
        simple_cut: If True, cuts holes to create simple polygons without holes
    
    Returns:
        Dictionary mapping label values to lists of Shapely polygons
    """
    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels != background_label]
    
    label_polygons = {}
    
    for label in unique_labels:
        binary_mask = (label_mask == label).astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary_mask, 
                                             cv2.RETR_CCOMP, 
                                             cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
            
        polygons = []
        hierarchy = hierarchy[0]
        
        # Group contours by parent-child relationship
        for i, (contour, h) in enumerate(zip(contours, hierarchy)):
            if h[3] == -1:  # External contour
                exterior = contour.squeeze()
                
                # Find all holes for this external contour
                holes = []
                for j, h_child in enumerate(hierarchy):
                    if h_child[3] == i:  # Hole belongs to current external
                        holes.append(contours[j].squeeze())
                
                try:
                    if holes:
                        # Create polygon with holes
                        poly = Polygon(exterior, holes=[hole for hole in holes])
                        if simple_cut:
                            poly = cut_polygon_recursive(poly)
                    else:
                        poly = Polygon(exterior)

                    # The polygons will necessarily have intersecting LinearRing for its exterior
                    # because of how we perform the cut, so we don't do the `is_valid` checking
                    polygons.append(poly)

                    # if poly.is_valid and poly.area > 0:
                    #     polygons.append(poly)
                    # else:
                    #     polygons.append(poly)
                    #     print("Polygon invalid because of", is_valid_reason(poly))
                except Exception as e:
                    print(f"Warning: Failed to create polygon for label {label}: {str(e)}")
                    continue
        
        if polygons:
            label_polygons[label] = polygons
    
    return label_polygons

def roi_dict_to_df(roi_dict):
    # Initialize lists to store data
    data = []
    current_id = 0

    # Iterate through the dictionary
    for dict_key, polygons in roi_dict.items():
        for idx, poly in enumerate(polygons):
            # Get coordinates from polygon exterior
            coords = list(poly.exterior.coords)
            # Format coordinates as string "x1,y1 x2,y2 ..."
            points_str = " ".join([f"{x},{y}" for x, y in coords])
            
            # Create row
            row = {
                'id': current_id,
                'Name': f"cluster_{dict_key}_roi_{idx}",
                'Text': "",
                'type': "Polygon",
                'all_points': points_str
            }
            data.append(row)
            current_id += 1

    # Create DataFrame
    df = pd.DataFrame(data)

    return df