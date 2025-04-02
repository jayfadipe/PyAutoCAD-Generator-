import traceback
import math 
import os   
import ast
from pyautocad import Autocad, APoint
# Removed problematic import: from pyautocad.contrib.conv import make_variant_array 
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
# IMPORTANT: User needs to create a .env file in this directory (PyAutoCAD_Generator)
# with the line: GOOGLE_API_KEY=YOUR_API_KEY_HERE

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = None # Initialize model variable

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    print("Please create a .env file in the PyAutoCAD_Generator directory with:")
    print("GOOGLE_API_KEY=YOUR_API_KEY_HERE")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Using the specific experimental model requested by the user
        model_name = "gemini-2.5-pro-exp-03-25" 
        print(f"Attempting to use model: {model_name}")
        model = genai.GenerativeModel(model_name) 
    except Exception as e:
        print(f"Error configuring Google Generative AI or initializing model '{model_name}': {e}")
        traceback.print_exc()

# --- Supported AutoCAD Entities (using pyautocad methods) ---
# Re-adding all requested entities based on documentation, using standard AddPolyline/AddSpline
SUPPORTED_ENTITIES = [
    "LINE", "CIRCLE", "ARC", "TEXT", "POINT", 
    "POLYLINE", "LWPOLYLINE", "RECTANGLE", "POLYGON", # Handled via AddPolyline (potential COM error)
    "SPLINE", # Re-added (potential COM error)
    "ELLIPSE", "SOLID", "XLINE", "RAY", "MTEXT", 
    "DONUT", # Re-added (workaround using AddPolyline - potential COM error)
    "MLINE", "TRACE", 
    "BOX", "SPHERE", "CYLINDER", "CONE", 
    "WEDGE", # Re-added (AddWedge might not exist/work)
    "TORUS", "PYRAMID", # 3D Solids (Pyramid as 3DFaces)
    "HATCH", # Basic support (boundary uses AddPolyline - potential COM error)
    "3DFACE", 
    "INSERT", # Basic support
    "DIMENSION" # Basic Linear/Aligned support
]
SUPPORTED_ENTITIES_STR = ", ".join(SUPPORTED_ENTITIES)


# --- Helper Functions ---

def calculate_polygon_vertices_3d(center: APoint, radius: float, num_sides: int) -> list:
    """Calculates 3D vertices for a regular polygon in the XY plane."""
    if num_sides < 3:
        return []
    angle_step = 2 * math.pi / num_sides
    vertices = []
    for i in range(num_sides):
        angle = i * angle_step
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        vertices.append([x, y, center.z]) 
    return vertices

# Helper function for Donut workaround
def create_donut_polyline_points(center, inner_radius, outer_radius, segments=32):
    """Generates points for two concentric circles to simulate a donut with polyline."""
    points = []
    # Outer circle
    for i in range(segments):
        angle = (2 * math.pi / segments) * i
        points.append([center.x + outer_radius * math.cos(angle), center.y + outer_radius * math.sin(angle), center.z])
    points.append(points[0]) # Close outer loop
    # Inner circle (reverse order)
    inner_start_index = len(points)
    for i in range(segments - 1, -1, -1):
        angle = (2 * math.pi / segments) * i
        points.append([center.x + inner_radius * math.cos(angle), center.y + inner_radius * math.sin(angle), center.z])
    points.append(points[inner_start_index]) # Close inner loop
    return points


def parse_llm_output(llm_string: str) -> list | None:
    """Safely parses the LLM string output into a Python list of dictionaries."""
    try:
        # Basic cleanup: remove potential markdown backticks
        if llm_string.startswith("```python"):
            llm_string = llm_string[len("```python"):].strip()
        if llm_string.startswith("```"):
             llm_string = llm_string[3:].strip()
        if llm_string.endswith("```"):
            llm_string = llm_string[:-3].strip()
            
        parsed_output = ast.literal_eval(llm_string)
        if isinstance(parsed_output, list):
            for item in parsed_output:
                if not isinstance(item, dict) or 'type' not in item:
                    print(f"Warning: Invalid item format in LLM output: {item}")
            return parsed_output 
        else:
            print(f"Error: LLM output is not a list: {parsed_output}")
            return None
    except (SyntaxError, ValueError, TypeError) as e:
        print(f"Error parsing LLM output: {e}\nOutput was:\n{llm_string}")
        return None

def connect_to_autocad():
    """Attempts to connect to a running AutoCAD instance."""
    try:
        print("Attempting to connect to AutoCAD...")
        acad = Autocad(create_if_not_exists=False) 
        doc_name = acad.doc.Name 
        print(f"Successfully connected to AutoCAD document: {doc_name}")
        return acad
    except Exception as e:
        print(f"Error connecting to AutoCAD: {e}")
        print("Please ensure a full version of AutoCAD is running AND a drawing is open.")
        traceback.print_exc()
        return None

def add_entities_to_drawing(acad, entities):
    """Adds entities to the AutoCAD drawing based on a list of dictionaries."""
    if not acad:
        print("Invalid AutoCAD connection object.")
        return
    try:
        if not acad.doc or not acad.model:
            print("Invalid AutoCAD document or modelspace.")
            return
    except Exception as access_e:
         print(f"Error accessing AutoCAD document/modelspace: {access_e}")
         print("AutoCAD might be busy or unresponsive.")
         return
         
    if not entities:
        print("No entities provided to add.")
        return

    space = acad.model 
    print(f"Targeting: Model Space")
    print(f"Adding {len(entities)} entities...")
    entity_count = 0

    for entity in entities:
        entity_type = entity.get('type', '').upper()
        try:
            added = False 
            if entity_type == 'LINE':
                start = APoint(entity.get('start', [0, 0, 0]))
                end = APoint(entity.get('end', [0, 0, 0]))
                space.AddLine(start, end)
                print(f"  Added LINE from {start} to {end}")
                added = True
            
            elif entity_type == 'CIRCLE':
                center = APoint(entity.get('center', [0, 0, 0]))
                radius = float(entity.get('radius', 1))
                space.AddCircle(center, radius)
                print(f"  Added CIRCLE at {center} with radius {radius}")
                added = True

            elif entity_type == 'ARC':
                center = APoint(entity.get('center', [0, 0, 0]))
                radius = float(entity.get('radius', 1))
                start_angle_deg = float(entity.get('start_angle', 0))
                end_angle_deg = float(entity.get('end_angle', 90))
                start_angle_rad = math.radians(start_angle_deg)
                end_angle_rad = math.radians(end_angle_deg)
                space.AddArc(center, radius, start_angle_rad, end_angle_rad)
                print(f"  Added ARC at {center}, radius {radius}, from {start_angle_deg} to {end_angle_deg} deg")
                added = True

            elif entity_type == 'TEXT':
                insert = APoint(entity.get('insert', [0, 0, 0]))
                text_string = str(entity.get('text', 'Default Text'))
                height = float(entity.get('height', 2.5))
                space.AddText(text_string, insert, height)
                print(f"  Added TEXT '{text_string}' at {insert} with height {height}")
                added = True

            elif entity_type == 'POINT':
                location = APoint(entity.get('location', [0, 0, 0]))
                space.AddPoint(location)
                print(f"  Added POINT at {location}")
                added = True

            elif entity_type in ['POLYLINE', 'LWPOLYLINE', 'RECTANGLE', 'POLYGON']: 
                # Using AddPolyline - Expect potential COM errors
                points_data = entity.get('points', []) 
                if entity_type == 'RECTANGLE' and not points_data:
                     corner1 = entity.get('corner1', [0,0,0])
                     corner2 = entity.get('corner2', [10,10,0])
                     x1, y1, z1 = corner1 + [0]*(3-len(corner1)) 
                     x2, y2, z2 = corner2 + [0]*(3-len(corner2)) 
                     points_data = [[x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1]]
                elif entity_type == 'POLYGON' and not points_data:
                     center_p = entity.get('center', [0,0,0])
                     radius_p = float(entity.get('radius', 5))
                     num_sides_p = int(entity.get('num_sides', 6))
                     if num_sides_p >= 3:
                         points_data = calculate_polygon_vertices_3d(APoint(center_p), radius_p, num_sides_p)
                     else:
                         print(f"  Warning: Skipping POLYGON with invalid sides: {num_sides_p}")
                         continue 
                
                if len(points_data) >= 2:
                    flat_points = []
                    for p in points_data:
                        p_3d = p + [0] * (3 - len(p)) if len(p) < 3 else p[:3]
                        flat_points.extend(p_3d)
                    
                    polyline_obj = space.AddPolyline(flat_points) # Using AddPolyline - might fail
                    is_closed = entity.get('closed', False) or entity_type in ['RECTANGLE', 'POLYGON']
                    if is_closed and len(points_data) > 2:
                        start_pt = APoint(flat_points[0], flat_points[1], flat_points[2])
                        end_pt = APoint(flat_points[-3], flat_points[-2], flat_points[-1])
                        if start_pt.distance_to(end_pt) > 1e-9: 
                             polyline_obj.AppendVertex(start_pt) 
                        try:
                            polyline_obj.Closed = True 
                        except Exception as poly_close_err:
                            print(f"  Info: Could not set Closed property for polyline: {poly_close_err}")
                    print(f"  Attempted POLYLINE/RECTANGLE/POLYGON with {len(points_data)} vertices. Closed: {is_closed}")
                    added = True
                else:
                    print(f"  Warning: Skipping {entity_type} due to insufficient points ({len(points_data)}). Needs >= 2.")

            elif entity_type == 'SPLINE': # Re-added - may cause COM error
                fit_points_data = entity.get('fit_points', [])
                if len(fit_points_data) >= 2:
                     flat_fit_points = []
                     for p in fit_points_data:
                         p_3d = p + [0] * (3 - len(p)) if len(p) < 3 else p[:3]
                         flat_fit_points.extend(p_3d)
                     
                     start_tangent = APoint(entity.get('start_tangent', [0,0,0])) 
                     end_tangent = APoint(entity.get('end_tangent', [0,0,0]))
                     
                     # Passing flat list - this might cause COM error again
                     space.AddSpline(flat_fit_points, start_tangent, end_tangent) 
                     print(f"  Attempted SPLINE with {len(fit_points_data)} fit points.")
                     added = True
                else:
                     print(f"  Warning: Skipping SPLINE due to insufficient fit points ({len(fit_points_data)}). Needs >= 2.")

            elif entity_type == 'ELLIPSE':
                center = APoint(entity.get('center', [0, 0, 0]))
                major_axis_pt = APoint(entity.get('major_axis_endpoint', [center.x+1, center.y, center.z])) 
                major_axis_vec = APoint(major_axis_pt.x - center.x, major_axis_pt.y - center.y, major_axis_pt.z - center.z)
                radius_ratio = float(entity.get('ratio', 0.5)) 
                space.AddEllipse(center, major_axis_vec, radius_ratio) 
                print(f"  Added ELLIPSE at {center} with major axis {major_axis_vec} and ratio {radius_ratio}")
                added = True

            elif entity_type == 'SOLID': 
                points_data = entity.get('points', [])
                if len(points_data) in [3, 4]:
                    points_ap = [APoint(p) for p in points_data]
                    p1, p2, p3 = points_ap[0], points_ap[1], points_ap[2]
                    p4 = points_ap[3] if len(points_ap) == 4 else p3 
                    space.AddSolid(p1, p2, p3, p4) 
                    print(f"  Added SOLID with {len(points_data)} points.")
                    added = True
                else:
                    print(f"  Warning: Skipping SOLID with invalid points ({len(points_data)}). Needs 3 or 4.")

            elif entity_type == 'XLINE':
                base_point = APoint(entity.get('base_point', [0, 0, 0]))
                direction_vec = APoint(entity.get('direction', [1, 0, 0])) 
                space.AddXLine(base_point, direction_vec) 
                print(f"  Added XLINE at {base_point} with direction {direction_vec}")
                added = True

            elif entity_type == 'RAY':
                start_point = APoint(entity.get('start_point', [0, 0, 0]))
                direction_vec = APoint(entity.get('direction', [1, 0, 0])) 
                space.AddRay(start_point, direction_vec) 
                print(f"  Added RAY from {start_point} with direction {direction_vec}")
                added = True

            elif entity_type == 'MTEXT':
                insert = APoint(entity.get('insert', [0, 0, 0]))
                text_string = str(entity.get('text', 'Default MText'))
                width = float(entity.get('width', 0)) 
                height = float(entity.get('height', 2.5)) 
                mtext_obj = space.AddMText(insert, width, text_string) 
                if height > 0: 
                    mtext_obj.Height = height 
                print(f"  Added MTEXT '{text_string[:20]}...' at {insert} with width {width}, height {height}")
                added = True

            elif entity_type == 'DONUT': # Workaround using Polyline - may cause COM error
                center = APoint(entity.get('center', [0, 0, 0]))
                inner_radius = float(entity.get('inner_radius', 1))
                outer_radius = float(entity.get('outer_radius', 2))
                if inner_radius < outer_radius and inner_radius >= 0:
                    donut_points = create_donut_polyline_points(center, inner_radius, outer_radius)
                    flat_donut_points = []
                    for p in donut_points:
                        flat_donut_points.extend(p)
                    donut_poly = space.AddPolyline(flat_donut_points) # Using AddPolyline - might fail
                    print(f"  Attempted DONUT (simulated as Polyline) at {center} with inner radius {inner_radius}, outer radius {outer_radius}")
                    added = True
                else:
                    print(f"  Warning: Skipping DONUT with invalid radii (inner={inner_radius}, outer={outer_radius}).")


            elif entity_type == 'MLINE':
                 vertices_data = entity.get('vertices', [])
                 if len(vertices_data) >= 2:
                     flat_vertices = []
                     for p in vertices_data:
                         p_3d = p + [0] * (3 - len(p)) if len(p) < 3 else p[:3]
                         flat_vertices.extend(p_3d)
                     space.AddMline(flat_vertices) 
                     print(f"  Added MLINE with {len(vertices_data)} vertices.")
                     added = True
                 else:
                     print(f"  Warning: Skipping MLINE due to insufficient vertices ({len(vertices_data)}). Needs >= 2.")

            elif entity_type == 'TRACE':
                 points_data = entity.get('points', [])
                 if len(points_data) >= 2:
                     flat_points = []
                     for p in points_data:
                         p_3d = p + [0] * (3 - len(p)) if len(p) < 3 else p[:3]
                         flat_points.extend(p_3d)
                     space.AddTrace(flat_points) 
                     print(f"  Added TRACE with {len(points_data)} points. (Width uses current TRACEWID setting)")
                     added = True
                 else:
                     print(f"  Warning: Skipping TRACE due to insufficient points ({len(points_data)}). Needs >= 2.")

            # --- 3D Solids ---
            elif entity_type == 'BOX':
                corner = APoint(entity.get('corner', [0,0,0])) 
                length = float(entity.get('length', entity.get('size', [10,10,10])[0])) 
                width = float(entity.get('width', entity.get('size', [10,10,10])[1]))  
                height = float(entity.get('height', entity.get('size', [10,10,10])[2])) 
                space.AddBox(corner, length, width, height) 
                print(f"  Added BOX at {corner} with L={length}, W={width}, H={height}")
                added = True

            elif entity_type == 'SPHERE':
                center = APoint(entity.get('center', [0,0,0]))
                radius = float(entity.get('radius', 5))
                space.AddSphere(center, radius) 
                print(f"  Added SPHERE at {center} with radius {radius}")
                added = True

            elif entity_type == 'CYLINDER':
                center = APoint(entity.get('center', [0,0,0])) 
                radius = float(entity.get('radius', 5))
                height = float(entity.get('height', 10))
                space.AddCylinder(center, radius, height) 
                print(f"  Added CYLINDER at {center} with radius {radius}, height {height}")
                added = True

            elif entity_type == 'CONE':
                center = APoint(entity.get('center', [0,0,0])) 
                base_radius = float(entity.get('radius', entity.get('base_radius', 5)))
                height = float(entity.get('height', 10))
                space.AddCone(center, base_radius, height) 
                print(f"  Added CONE at {center} with base radius {base_radius}, height {height}")
                added = True

            elif entity_type == 'TORUS':
                center = APoint(entity.get('center', [0,0,0]))
                torus_radius = float(entity.get('major_radius', entity.get('torus_radius', 10))) 
                tube_radius = float(entity.get('minor_radius', entity.get('tube_radius', 1)))   
                space.AddTorus(center, torus_radius, tube_radius) 
                print(f"  Added TORUS at {center} with torus radius {torus_radius}, tube radius {tube_radius}")
                added = True

            elif entity_type == 'PYRAMID':
                center = APoint(entity.get('center', [0,0,0])) 
                side = float(entity.get('side_length', 10))
                height = float(entity.get('height', 10))
                half_side = side / 2.0
                p1 = APoint(center.x - half_side, center.y - half_side, center.z)
                p2 = APoint(center.x + half_side, center.y - half_side, center.z)
                p3 = APoint(center.x + half_side, center.y + half_side, center.z)
                p4 = APoint(center.x - half_side, center.y + half_side, center.z)
                apex = APoint(center.x, center.y, center.z + height)
                space.Add3DFace(p1, p2, apex) 
                space.Add3DFace(p2, p3, apex) 
                space.Add3DFace(p3, p4, apex) 
                space.Add3DFace(p4, p1, apex) 
                space.Add3DFace(p1, p2, p3, p4) 
                print(f"  Added PYRAMID (as 3DFaces) centered near {center} with base side {side}, height {height}")
                added = True
                
            elif entity_type == 'WEDGE': # Re-added attempt
                 corner = APoint(entity.get('corner', [0,0,0]))
                 length = float(entity.get('length', entity.get('size', [10,10,10])[0])) 
                 width = float(entity.get('width', entity.get('size', [10,10,10])[1]))  
                 height = float(entity.get('height', entity.get('size', [10,10,10])[2])) 
                 try:
                     space.AddWedge(corner, length, width, height) 
                     print(f"  Added WEDGE at {corner} with L={length}, W={width}, H={height}")
                     added = True
                 except AttributeError:
                     print(f"  Warning: Skipping WEDGE. AddWedge method not found in this pyautocad version.")
                     added = True # Mark as processed to avoid unsupported message
                 except Exception as wedge_err:
                     print(f"  Error adding WEDGE: {wedge_err}")


            elif entity_type == 'HATCH': # Uses AddPolyline - may cause COM error
                 points_data = entity.get('boundary_points', [])
                 pattern_name = str(entity.get('pattern_name', 'SOLID')) 
                 associativity = bool(entity.get('associativity', True))
                 if len(points_data) >= 3:
                     flat_points = []
                     for p in points_data:
                         p_3d = p + [0] * (3 - len(p)) if len(p) < 3 else p[:3]
                         flat_points.extend(p_3d)
                     
                     temp_boundary = space.AddPolyline(flat_points) # Using AddPolyline - might fail
                     
                     start_pt = APoint(flat_points[0], flat_points[1], flat_points[2])
                     end_pt = APoint(flat_points[-3], flat_points[-2], flat_points[-1])
                     if start_pt.distance_to(end_pt) > 1e-9:
                          temp_boundary.AppendVertex(start_pt)
                     try:
                         temp_boundary.Closed = True
                     except Exception as poly_close_err:
                         print(f"  Info: Could not set Closed property for hatch boundary: {poly_close_err}")

                     pattern_type = 1 
                     try:
                         hatch_obj = space.AddHatch(pattern_type, pattern_name, associativity, (temp_boundary,)) 
                         print(f"  Attempted HATCH with pattern '{pattern_name}' using {len(points_data)} boundary points.")
                         added = True
                     except Exception as hatch_err:
                          print(f"  Error creating HATCH: {hatch_err}")
                          try:
                              temp_boundary.Delete() 
                          except: pass 
                 else:
                     print(f"  Warning: Skipping HATCH due to insufficient boundary points ({len(points_data)}). Needs >= 3.")

            elif entity_type == '3DFACE': 
                 points_data = entity.get('points', [])
                 if len(points_data) in [3, 4]:
                     points_ap = [APoint(p) for p in points_data]
                     p1, p2, p3 = points_ap[0], points_ap[1], points_ap[2]
                     p4 = points_ap[3] if len(points_ap) == 4 else p3
                     space.Add3DFace(p1, p2, p3, p4) 
                     print(f"  Added 3DFACE with {len(points_data)} points.")
                     added = True
                 else:
                     print(f"  Warning: Skipping 3DFACE with invalid points ({len(points_data)}). Needs 3 or 4.")

            elif entity_type == 'REGION':
                 print(f"  Warning: Skipping REGION. Generating regions from text is complex and not supported.")
                 added = True 

            elif entity_type == 'INSERT': 
                 block_name = entity.get('block_name')
                 insert_point = APoint(entity.get('insert', [0, 0, 0]))
                 x_scale = float(entity.get('x_scale', 1.0))
                 y_scale = float(entity.get('y_scale', 1.0))
                 z_scale = float(entity.get('z_scale', 1.0))
                 rotation_deg = float(entity.get('rotation', 0.0))
                 rotation_rad = math.radians(rotation_deg)
                 if block_name:
                     try:
                        space.InsertBlock(insert_point, block_name, x_scale, y_scale, z_scale, rotation_rad) 
                        print(f"  Added INSERT for block '{block_name}' at {insert_point}")
                        added = True
                     except Exception as insert_err:
                         print(f"  Error inserting block '{block_name}' (ensure it's defined in the drawing): {insert_err}")
                 else:
                     print(f"  Warning: Skipping INSERT due to missing 'block_name'.")

            elif entity_type == 'DIMENSION':
                 dim_type = entity.get('dim_type', 'LINEAR').upper()
                 p1 = APoint(entity.get('p1', [0,0,0])) 
                 p2 = APoint(entity.get('p2', [10,0,0])) 
                 location = APoint(entity.get('location', [5, 5, 0])) 
                 if dim_type == 'LINEAR':
                      rotation_deg = float(entity.get('rotation', 0.0)) 
                      rotation_rad = math.radians(rotation_deg)
                      space.AddDimLinear(p1, p2, location, rotation_rad) 
                      print(f"  Added LINEAR DIMENSION between {p1} and {p2} at {location}")
                      added = True
                 elif dim_type == 'ALIGNED':
                      space.AddDimAligned(p1, p2, location) 
                      print(f"  Added ALIGNED DIMENSION between {p1} and {p2} at {location}")
                      added = True
                 else:
                      print(f"  Warning: Unsupported DIMENSION type '{dim_type}'. Only LINEAR and ALIGNED are implemented.")
            
            # --- Final Check for Unsupported Types ---
            if not added:
                print(f"  Warning: Unsupported entity type '{entity_type}' encountered. Skipping.")

        except Exception as e_entity:
            print(f"  Error adding entity {entity_type}: {e_entity}. Entity data: {entity}")
            traceback.print_exc() 
        
        if added:
            entity_count += 1

    print(f"Finished adding entities. {entity_count} entities processed.")


def main():
    """Main function to connect, add entities, and save."""
    
    # --- LLM Setup ---
    if not model:
        print("LLM Model not initialized. Please check API Key and configuration.")
        return
        
    # --- Get User Input ---
    user_description = input("Enter CAD description: ")
    if not user_description:
        print("No description entered.")
        return

    # --- Connect to AutoCAD ---
    acad = connect_to_autocad()
    if not acad:
        return 

    # --- LLM Interaction ---
    llm_output_verified = None
    entities_verified = None
    try:
        # --- LLM Call 1: Initial Generation ---
        prompt1 = f"""
Based on the following user description, generate a Python list of dictionaries representing CAD entities for AutoCAD using pyautocad.
Each dictionary MUST include a 'type' key. Choose the most appropriate type from the supported list: {SUPPORTED_ENTITIES_STR}. For example, if the user asks for a 'square' or 'rectangle', use 'RECTANGLE'; if they ask for a 'smooth curve', use 'SPLINE'. If the specific type is mentioned (e.g., 'draw a LINE'), use that type.
Include all necessary parameters for the chosen 'type' based on standard AutoCAD practices and the pyautocad library requirements (e.g., 'start'/'end' for LINE; 'center'/'radius' for CIRCLE; 'points' for 3DFACE/SOLID/TRACE/POLYLINE; 'insert'/'text'/'height' for TEXT; 'insert'/'text'/'width' for MTEXT; 'boundary_points' for HATCH; 'corner1'/'corner2' for RECTANGLE; 'center'/'radius'/'num_sides' for POLYGON; 'center'/'inner_radius'/'outer_radius' for DONUT; 'corner'/'length'/'width'/'height' for BOX; 'center'/'radius' for SPHERE; 'center'/'radius'/'height' for CYLINDER/CONE; 'center'/'torus_radius'/'tube_radius' for TORUS; 'center'/'side_length'/'height' for PYRAMID; 'vertices' for MLINE; 'block_name'/'insert' for INSERT; 'dim_type'/'p1'/'p2'/'location' for DIMENSION).
IMPORTANT NOTES: 
- POLYLINE, LWPOLYLINE, RECTANGLE, POLYGON are created using AddPolyline (potential COM error).
- SPLINE is created using AddSpline (potential COM error).
- DONUT is simulated using AddPolyline (potential COM error).
- HATCH boundary uses AddPolyline (potential COM error). Only 'SOLID' pattern_name is reliably supported.
- WEDGE support is experimental (AddWedge may not exist).
- For INSERT, the block_name must refer to a block already defined in the target AutoCAD drawing. Block definitions cannot be created here.
- For DIMENSION, only 'LINEAR' and 'ALIGNED' dim_types are supported using the default style.
- 3D shapes (BOX, SPHERE, etc.) are generated as true 3D Solids where possible (BOX, SPHERE, CYLINDER, CONE, TORUS). PYRAMID is made of 3DFaces.
Ensure coordinates and all other numerical parameters are lists or numbers as appropriate (e.g., [10.5, 20, 0] or 25.0), NOT arithmetic expressions (e.g., NOT [100/2, 4*5, 0]).
Be precise with measurements and coordinates mentioned in the description.
Output ONLY the Python list of dictionaries, without any surrounding text or explanations.

Example 1:
User Description: "Draw a 50x30 rectangle starting at 10,10."
Output: [{{'type': 'RECTANGLE', 'corner1': [10, 10, 0], 'corner2': [60, 40, 0]}}]

Example 2:
User Description: "A smooth curve through points (0,0), (5,5), (10,0)."
Output: [{{'type': 'SPLINE', 'fit_points': [[0, 0, 0], [5, 5, 0], [10, 0, 0]]}}]

User Description:
"{user_description}"
"""
        print("\n--- Sending Prompt 1 to LLM ---")
        response1 = model.generate_content(prompt1)
        entities_str_initial = response1.text
        print(f"--- Received Response 1 from LLM ---\n{entities_str_initial}")

        entities_initial = parse_llm_output(entities_str_initial)

        if not entities_initial:
            print("Error: Failed to parse the initial LLM output. The format might be incorrect.")
            return

        # --- LLM Call 2: Verification ---
        prompt2 = f"""
You are a CAD expert reviewing an LLM-generated list of entities based on a user request for pyautocad.
Original User Request: "{user_description}"
Generated Entity List (Python):
{entities_initial}

Review the generated list for accuracy, completeness, and adherence to standard AutoCAD entity parameters compatible with pyautocad.
Supported entity types are: {SUPPORTED_ENTITIES_STR}.
Ensure all necessary parameters for each entity type are present and correctly formatted (e.g., coordinates as lists of numbers).
Correct any mistakes, omissions, or formatting errors found in the list based *only* on the original user request and the supported types/parameters. Remove any entities with types not in the supported list.
Output ONLY the final, verified Python list of dictionaries, without any surrounding text or explanations. If the list is already perfect, output it as is.
"""
        print("\n--- Sending Prompt 2 to LLM (Verification) ---")
        response2 = model.generate_content(prompt2)
        entities_str_verified = response2.text
        llm_output_verified = entities_str_verified # Store for display (optional)
        print(f"--- Received Response 2 from LLM (Verified) ---\n{entities_str_verified}")

        entities_verified = parse_llm_output(entities_str_verified)

        if not entities_verified:
            print("Error: Failed to parse the verified LLM output. The format might be incorrect.")
            return

    except Exception as e:
        print(f"\nAn error occurred during LLM interaction: {e}")
        traceback.print_exc()
        return

    # --- Add Entities to Drawing ---
    add_entities_to_drawing(acad, entities_verified)

    # --- Save the Drawing ---
    try:
        save_path = os.path.join(os.path.expanduser("~"), "Desktop", "pyautocad_llm_output.dwg") 
        acad.doc.SaveAs(save_path)
        print(f"\nDrawing saved successfully to: {save_path}")
    except Exception as e:
        print(f"\nError saving drawing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
