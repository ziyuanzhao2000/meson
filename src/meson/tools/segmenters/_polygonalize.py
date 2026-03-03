import numpy as np
import cv2
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, LinearRing, LineString
from shapely.geometry.polygon import orient
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


def _perpendicular_offset(p1: Tuple[float, float], p2: Tuple[float, float],
                          epsilon: float) -> Tuple[float, float]:
    """
    Compute a perpendicular offset vector of magnitude `epsilon` for the
    segment p1->p2.  Returns (dx, dy) to be *added* to a point so that it
    shifts to one side of the segment.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.hypot(dx, dy)
    if length == 0:
        return (epsilon, 0.0)          # degenerate – just nudge in x
    # unit perpendicular (rotate 90° CCW)
    perp_x = -dy / length * epsilon
    perp_y =  dx / length * epsilon
    return (perp_x, perp_y)

# def cut_polygon_recursive(polygon: Polygon, epsilon: float = 0.5,
#                           _depth: int = 0, _max_depth: int = 100) -> Polygon:
#     if _depth >= _max_depth:
#         return polygon

#     if len(polygon.interiors) == 0:
#         return polygon

#     # Normalize: ensure exterior is CCW and holes are CW
#     polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)

#     first_hole = polygon.interiors[0]
#     other_holes = list(polygon.interiors[1:])

#     nearest_ext, nearest_hole, ext_idx, hole_idx = find_nearest_vertices(
#         polygon.exterior, first_hole
#     )

#     off = _perpendicular_offset(nearest_ext, nearest_hole, epsilon)

#     ext_out  = (nearest_ext[0]  - off[0], nearest_ext[1]  - off[1])
#     hole_out = (nearest_hole[0] - off[0], nearest_hole[1] - off[1])
#     ext_ret  = (nearest_ext[0]  + off[0], nearest_ext[1]  + off[1])
#     hole_ret = (nearest_hole[0] + off[0], nearest_hole[1] + off[1])

#     # After orient(), exterior is CCW → traverse in stored order
#     exterior_vertices = get_ordered_vertices(polygon.exterior, ext_idx, True)
#     # After orient(), holes are CW → traverse in stored order
#     hole_vertices = get_ordered_vertices(first_hole, hole_idx, True)

#     new_coords = []
#     new_coords.extend(exterior_vertices)
#     new_coords.append(ext_out)
#     new_coords.append(hole_out)
#     new_coords.extend(hole_vertices)
#     new_coords.append(hole_ret)
#     new_coords.append(ext_ret)

#     new_polygon = Polygon(new_coords, [list(hole.coords) for hole in other_holes])

#     return cut_polygon_recursive(new_polygon, epsilon=epsilon * 0.8,
#                                  _depth=_depth + 1, _max_depth=_max_depth)

def cut_polygon_clean(polygon, epsilon=1.0, overshoot=2.0):
    polygon = orient(polygon, sign=1.0)
    
    while len(polygon.interiors) > 0:
        hole = polygon.interiors[0]
        
        nearest_ext, nearest_hole, _, _ = find_nearest_vertices(
            polygon.exterior, hole
        )
        
        cut_line = LineString([nearest_ext, nearest_hole])
        corridor = cut_line.buffer(epsilon, cap_style=1)
        polygon = polygon.difference(corridor)
        
        # Extract the largest polygon from whatever geometry type comes back
        polygon = _extract_largest_polygon(polygon)
        
        if polygon is None:
            return Polygon()  # completely degenerate
    
    return polygon


def _extract_largest_polygon(geom):
    """Extract the largest Polygon from any geometry type."""
    if isinstance(geom, Polygon):
        return geom
    
    # Collect all polygons from MultiPolygon or GeometryCollection
    polygons = []
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        for part in geom.geoms:
            if isinstance(part, Polygon) and part.area > 0:
                polygons.append(part)
    
    if polygons:
        return max(polygons, key=lambda g: g.area)
    
    return None

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
                            poly = cut_polygon_clean(poly)
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