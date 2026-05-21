import os
import pandas as pd
import numpy as np
import copy
from .abstract_solver import AbstractSolver

class solver_362540_362159_354745_360043(AbstractSolver):
    def __init__(self, inst):
        super().__init__(inst)
        self.name = 'solver_362540_362159_354745_360043'

    def get_item_dims(self, item, orient):
        w, d, h = item["width"], item["depth"], item["height"]
        rotations = [
            (w, d, h), (d, w, h), (h, d, w), (d, h, w), (w, h, d), (h, w, d),
        ]
        return rotations[orient]

    def overlap_1d(self, a_min, a_max, b_min, b_max):
        return max(a_min, b_min) < min(a_max, b_max)

    def boxes_overlap(self, b1, b2):
        return (self.overlap_1d(b1["x1"], b1["x2"], b2["x1"], b2["x2"]) and
                self.overlap_1d(b1["y1"], b1["y2"], b2["y1"], b2["y2"]) and
                self.overlap_1d(b1["z1"], b1["z2"], b2["z1"], b2["z2"]))

    def solve(self):
        print(f"Solving instance: {self.inst.name}")
        
        strategies = [
            ('volume', ['volume', 'weight']),
            ('area', ['footprint', 'volume']),
            ('height', ['height', 'volume'])
        ]
        
        best_cost = float('inf')
        best_sol = None

        for strategy_name, sort_cols in strategies:
            items_df = self.inst.df_items.copy()
            items_df['volume'] = items_df['width'] * items_df['depth'] * items_df['height']
            items_df['footprint'] = items_df['width'] * items_df['depth']
            items_df = items_df.sort_values(by=sort_cols, ascending=False)
            
            current_sol, current_cost = self.run_packing_dynamic(items_df)
            
            print(f"  Strategy {strategy_name}: Cost {current_cost:.2f}")
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_sol = current_sol
        
        self.sol = best_sol
        print(f"Best strategy found. Total cost: {best_cost:.2f}")
        self.write_solution_to_file()

    def run_packing_dynamic(self, items_df):
        vehicles_info = self.inst.df_vehicles.to_dict('index')
        for v_type in vehicles_info:
            v = vehicles_info[v_type]
            if pd.isna(v.get('maxValue')):
                v['maxValue'] = float('inf')
        
        remaining_items = items_df.copy()
        final_solution = {
            'type_vehicle': [], 'idx_vehicle': [], 'id_item': [],
            'x_origin': [], 'y_origin': [], 'z_origin': [], 'orient': []
        }
        global_v_idx = 0
        
        while not remaining_items.empty:
            best_v_type = None
            best_v_sol_segment = []
            best_v_packed_ids = []
            best_v_cost_eff = float('inf')
            
            # Point 3: Look-ahead Vehicle Selection
            # Try every vehicle type and see which one handles the current items best
            for v_type, v_info in vehicles_info.items():
                packed_ids, sol_segment = self.simulate_vehicle_packing(v_type, v_info, remaining_items)
                
                if not packed_ids:
                    continue
                
                packed_vol = remaining_items.loc[packed_ids, 'volume'].sum()
                cost_eff = v_info['cost'] / packed_vol
                
                if cost_eff < best_v_cost_eff:
                    best_v_cost_eff = cost_eff
                    best_v_type = v_type
                    best_v_sol_segment = sol_segment
                    best_v_packed_ids = packed_ids
            
            if best_v_type:
                # Commit the best vehicle
                for item_id, x, y, z, orient in best_v_sol_segment:
                    final_solution['type_vehicle'].append(best_v_type)
                    final_solution['idx_vehicle'].append(global_v_idx)
                    final_solution['id_item'].append(item_id)
                    final_solution['x_origin'].append(x)
                    final_solution['y_origin'].append(y)
                    final_solution['z_origin'].append(z)
                    final_solution['orient'].append(orient)
                
                global_v_idx += 1
                remaining_items = remaining_items.drop(best_v_packed_ids)
            else:
                print(f"Warning: Could not place remaining {len(remaining_items)} items.")
                break
                
        total_cost = 0
        if final_solution['type_vehicle']:
            used_indices = set(final_solution['idx_vehicle'])
            for idx in used_indices:
                # Find the type of this vehicle
                v_type = final_solution['type_vehicle'][final_solution['idx_vehicle'].index(idx)]
                total_cost += vehicles_info[v_type]['cost']
                
        return final_solution, total_cost

    def calculate_contact_score(self, box, v_info, items):
        score = 0
        # Wall contacts
        if abs(box["x1"]) < 1e-7: score += (box["y2"] - box["y1"]) * (box["z2"] - box["z1"])
        if abs(box["x2"] - v_info["depth"]) < 1e-7: score += (box["y2"] - box["y1"]) * (box["z2"] - box["z1"])
        if abs(box["y1"]) < 1e-7: score += (box["x2"] - box["x1"]) * (box["z2"] - box["z1"])
        if abs(box["y2"] - v_info["width"]) < 1e-7: score += (box["x2"] - box["x1"]) * (box["z2"] - box["z1"])
        if abs(box["z1"]) < 1e-7: score += (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
        if abs(box["z2"] - v_info["height"]) < 1e-7: score += (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])

        # Item contacts
        for other in items:
            # Check for flush faces in all 3 dimensions
            # X-faces
            if abs(box["x1"] - other["x2"]) < 1e-7 or abs(box["x2"] - other["x1"]) < 1e-7:
                ix = 0 # Not overlapping in X, but flush
                iy = max(0, min(box["y2"], other["y2"]) - max(box["y1"], other["y1"]))
                iz = max(0, min(box["z2"], other["z2"]) - max(box["z1"], other["z1"]))
                score += iy * iz
            # Y-faces
            if abs(box["y1"] - other["y2"]) < 1e-7 or abs(box["y2"] - other["y1"]) < 1e-7:
                ix = max(0, min(box["x2"], other["x2"]) - max(box["x1"], other["x1"]))
                iy = 0
                iz = max(0, min(box["z2"], other["z2"]) - max(box["z1"], other["z1"]))
                score += ix * iz
            # Z-faces
            if abs(box["z1"] - other["z2"]) < 1e-7 or abs(box["z2"] - other["z1"]) < 1e-7:
                ix = max(0, min(box["x2"], other["x2"]) - max(box["x1"], other["x1"]))
                iy = max(0, min(box["y2"], other["y2"]) - max(box["y1"], other["y1"]))
                iz = 0
                score += ix * iy
        return score

    def simulate_vehicle_packing(self, v_type, v_info, items_df):
        v_data = {
            'type': v_type,
            'items': [],
            'total_weight': 0,
            'total_value': 0,
            'eps': [(0, 0, 0)]
        }
        
        packed_ids = []
        sol_segment = []
        
        for item_id, item in items_df.iterrows():
            if v_data['total_weight'] + item.weight > v_info['maxWeight'] + 1e-6:
                continue
            if v_data['total_value'] + item.value > v_info['maxValue'] + 1e-6:
                continue

            allowed_rots = [int(r) for r in str(item.allowedRotations)]
            rot_candidates = []
            for rot in allowed_rots:
                w_rot, d_rot, h_rot = self.get_item_dims(item, rot)
                rot_candidates.append((rot, w_rot, d_rot, h_rot))
            
            # Point 6: Rotation Optimization
            rot_candidates.sort(key=lambda x: (x[3], -(x[1]*x[2])))

            # Point 2: EP Selection - Sort by Z, then Y, then X for baseline preference
            v_data['eps'].sort(key=lambda p: (p[2], p[1], p[0]))

            best_placement = None
            max_score = -1

            for ep in v_data['eps']:
                ex, ey, ez = ep
                for rot, w_rot, d_rot, h_rot in rot_candidates:
                    box = {
                        "x1": ex, "y1": ey, "z1": ez,
                        "x2": ex + d_rot, "y2": ey + w_rot, "z2": ez + h_rot,
                        "base_area": w_rot * d_rot
                    }

                    if box["x2"] > v_info["depth"] + 1e-7 or \
                       box["y2"] > v_info["width"] + 1e-7 or \
                       box["z2"] > v_info["height"] + 1e-7:
                        continue
                    
                    overlap = False
                    for other in v_data['items']:
                        if self.boxes_overlap(box, other):
                            overlap = True
                            break
                    if overlap: continue

                    # Gravity check
                    if ez > 1e-7:
                        support_area = 0
                        for other in v_data['items']:
                            if abs(other["z2"] - ez) < 1e-6:
                                ix = max(0, min(box["x2"], other["x2"]) - max(box["x1"], other["x1"]))
                                iy = max(0, min(box["y2"], other["y2"]) - max(box["y1"], other["y1"]))
                                support_area += ix * iy
                        if support_area < box["base_area"] * (v_info["gravityStrength"] / 100.0) - 1e-7:
                            continue

                    # Calculate score (Corner Fit / Touching Surface Area)
                    score = self.calculate_contact_score(box, v_info, v_data['items'])
                    
                    # Tie-breaker: prefer lower coordinates if scores are equal
                    if score > max_score:
                        max_score = score
                        best_placement = (ep, rot, box, item_id)
            
            if best_placement:
                ep, rot, box, item_id = best_placement
                ex, ey, ez = ep
                
                # Success
                v_data['items'].append(box)
                v_data['total_weight'] += item.weight
                v_data['total_value'] += item.value
                packed_ids.append(item_id)
                sol_segment.append((item_id, ex, ey, ez, rot))
                
                # Point 4: EP Update
                v_data['eps'].remove(ep)
                v_data['eps'] = [p for p in v_data['eps'] if not (box['x1'] <= p[0] < box['x2'] - 1e-7 and 
                                                                    box['y1'] <= p[1] < box['y2'] - 1e-7 and 
                                                                    box['z1'] <= p[2] < box['z2'] - 1e-7)]
                
                new_pts = [(box['x2'], ey, ez), (ex, box['y2'], ez), (ex, ey, box['z2'])]
                for np_ in new_pts:
                    if np_[0] < v_info['depth'] - 1e-7 and np_[1] < v_info['width'] - 1e-7 and np_[2] < v_info['height'] - 1e-7:
                        covered = False
                        for b in v_data['items']:
                            if b['x1'] <= np_[0] < b['x2'] - 1e-7 and \
                                b['y1'] <= np_[1] < b['y2'] - 1e-7 and \
                                b['z1'] <= np_[2] < b['z2'] - 1e-7:
                                covered = True
                                break
                        if not covered and np_ not in v_data['eps']:
                            v_data['eps'].append(np_)
        
        return packed_ids, sol_segment
