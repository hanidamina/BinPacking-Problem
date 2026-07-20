import pandas as pd
import os
import time
import random
import concurrent.futures
from .abstract_solver import AbstractSolver

# Global functions for multiprocessing pickling
def get_dx_dy_dz(item, orient):
    w, d, h = item["width"], item["depth"], item["height"]
    rotations = [
        (w, d, h), (d, w, h), (h, d, w),
        (d, h, w), (w, h, d), (h, w, d),
    ]
    rw, rd, rh = rotations[orient]
    return rd, rw, rh

def find_best_placement(item, vehicle, eps, placed, ep_mode, orient_mode):
    if ep_mode == 'BFL':
        sorted_eps = sorted(eps, key=lambda p: (p[2], p[0], p[1]))
    elif ep_mode == 'BLF':
        sorted_eps = sorted(eps, key=lambda p: (p[2], p[1], p[0]))
    elif ep_mode == 'FBL':
        sorted_eps = sorted(eps, key=lambda p: (p[0], p[2], p[1]))
    elif ep_mode == 'FLB':
        sorted_eps = sorted(eps, key=lambda p: (p[0], p[1], p[2]))
    elif ep_mode == 'LBF':
        sorted_eps = sorted(eps, key=lambda p: (p[1], p[2], p[0]))
    else:
        sorted_eps = sorted(eps, key=lambda p: (p[2], p[0], p[1]))
        
    v_depth, v_width, v_height = vehicle['depth'], vehicle['width'], vehicle['height']
    v_gravity = vehicle['gravityStrength'] / 100.0
    
    for p in sorted_eps:
        x, y, z = p
        
        valid_orients = []
        for orient in item['allowed_list']:
            dx, dy, dz = get_dx_dy_dz(item, orient)
            if x + dx > v_depth or y + dy > v_width or z + dz > v_height:
                continue
            
            x2, y2, z2 = x + dx, y + dy, z + dz
            overlap = False
            for b in placed:
                if not (x2 <= b['x1'] or x >= b['x2'] or \
                        y2 <= b['y1'] or y >= b['y2'] or \
                        z2 <= b['z1'] or z >= b['z2']):
                    overlap = True
                    break
            if overlap: continue
            
            if z > 0:
                sup = 0
                for b in placed:
                    if abs(b['z2'] - z) < 1e-7:
                        inter_dx = max(0, min(x2, b['x2']) - max(x, b['x1']))
                        inter_dy = max(0, min(y2, b['y2']) - max(y, b['y1']))
                        sup += inter_dx * inter_dy
                if sup < v_gravity * (dx * dy) - 1e-9:
                    continue
            
            if orient_mode == 'first':
                return (x, y, z, orient, dx, dy, dz)
            valid_orients.append((orient, dx, dy, dz))
            
        if valid_orients:
            # Heuristic: minimize Z2, then X2, then Y2
            valid_orients.sort(key=lambda o: (z + o[3], x + o[1], y + o[2]))
            best_o, b_dx, b_dy, b_dz = valid_orients[0]
            return (x, y, z, best_o, b_dx, b_dy, b_dz)
            
    return None

def filter_eps(eps, vehicle, placed):
    res = []
    v_d, v_w, v_h = vehicle['depth'], vehicle['width'], vehicle['height']
    for x, y, z in eps:
        if x >= v_d or y >= v_w or z >= v_h: continue
        inside = False
        for b in placed:
            if x >= b['x1'] and x < b['x2'] and \
               y >= b['y1'] and y < b['y2'] and \
               z >= b['z1'] and z < b['z2']:
                inside = True
                break
        if not inside: res.append((x, y, z))
    return res

def pack_one_vehicle(vehicle, items, ep_mode, orient_mode):
    placed_boxes = []
    res_sol = []
    used_indices = []
    curr_w = 0
    curr_v = 0
    eps = [(0, 0, 0)]
    max_val = vehicle['max_val']
    max_weight = vehicle['maxWeight']
    
    for idx, item in enumerate(items):
        if curr_w + item['weight'] > max_weight: continue
        if curr_v + item['value'] > max_val: continue
        
        best_p = find_best_placement(item, vehicle, eps, placed_boxes, ep_mode, orient_mode)
        if best_p:
            x, y, z, orient, dx, dy, dz = best_p
            res_sol.append((item['id'], x, y, z, orient))
            placed_boxes.append({
                'x1': x, 'y1': y, 'z1': z,
                'x2': x + dx, 'y2': y + dy, 'z2': z + dz,
                'area': dx * dy
            })
            used_indices.append(idx)
            curr_w += item['weight']
            curr_v += item['value']
            eps.extend([(x+dx, y, z), (x, y+dy, z), (x, y, z+dz)])
            eps = filter_eps(set(eps), vehicle, placed_boxes)
    return res_sol, used_indices

def run_greedy_pack_worker(items, v_types, sort_mode, ep_mode, orient_mode, seed=None):
    if seed is not None: random.seed(seed)
    
    if sort_mode == 'vol':
        sorted_items = sorted(items, key=lambda x: (x['volume'], x['height'], x['width']), reverse=True)
    elif sort_mode == 'vol_w':
        sorted_items = sorted(items, key=lambda x: (x['volume'], x['height'], x['weight']), reverse=True)
    elif sort_mode == 'height':
        sorted_items = sorted(items, key=lambda x: (x['height'], x['volume'], x['width']), reverse=True)
    elif sort_mode == 'footprint':
        sorted_items = sorted(items, key=lambda x: (x['width'] * x['depth'], x['height'], x['volume']), reverse=True)
    elif sort_mode == 'weight':
        sorted_items = sorted(items, key=lambda x: (x['weight'], x['volume'], x['height']), reverse=True)
    elif sort_mode == 'val_dens':
        sorted_items = sorted(items, key=lambda x: (x['value'] / x['volume'] if x['volume'] > 0 else 0, x['volume']), reverse=True)
    elif sort_mode == 'random':
        sorted_items = list(items); random.shuffle(sorted_items)
    elif sort_mode.startswith('p_'):
        sorted_items = sorted(items, key=lambda x: x['volume'] * random.uniform(0.7, 1.3), reverse=True)
    else:
        sorted_items = items
    
    remaining = list(sorted_items)
    vehicle_idx = 0
    all_placements = []
    total_cost = 0
    
    while remaining:
        best_v_type = None
        best_packed = None
        best_indices = None
        max_eff = -1
        
        for v_info in v_types:
            packed, indices = pack_one_vehicle(v_info, remaining, ep_mode, orient_mode)
            if packed:
                packed_vol = sum(remaining[i]['volume'] for i in indices)
                eff = packed_vol / v_info['cost']
                if eff > max_eff:
                    max_eff = eff
                    best_v_type = v_info['type']; best_packed = packed
                    best_indices = indices; best_v_cost = v_info['cost']
        
        if not best_packed:
            v_info = v_types[0]; item = remaining[0]; orient = item['allowed_list'][0]
            best_v_type = v_info['type']; best_packed = [(item['id'], 0, 0, 0, orient)]
            best_indices = [0]; best_v_cost = v_info['cost']
            
        for item_id, x, y, z, orient in best_packed:
            all_placements.append({
                'type_vehicle': best_v_type, 'idx_vehicle': vehicle_idx, 'id_item': item_id,
                'x_origin': x, 'y_origin': y, 'z_origin': z, 'orient': orient
            })
        total_cost += best_v_cost
        used_set = set(best_indices)
        remaining = [it for idx, it in enumerate(remaining) if idx not in used_set]
        vehicle_idx += 1
    return all_placements, total_cost

class solver_362540(AbstractSolver):
    def __init__(self, inst):
        super().__init__(inst); self.name = 'solver_362540'

    def solve(self):
        start_time = time.time()
        items_dict = self.inst.df_items.to_dict('index')
        vehicles_dict = self.inst.df_vehicles.to_dict('index')
        
        items = []
        for iid, it in items_dict.items():
            it['id'] = iid
            it['volume'] = it['width'] * it['depth'] * it['height']
            it['allowed_list'] = [int(c) for c in str(it['allowedRotations'])]
            items.append(it)
            
        v_types = []
        for vtype, v in vehicles_dict.items():
            v['type'] = vtype; v['volume'] = v['width'] * v['depth'] * v['height']
            v['efficiency'] = v['volume'] / v['cost']
            v['max_val'] = v['maxValue'] if pd.notnull(v['maxValue']) else float('inf')
            v_types.append(v)
        v_types.sort(key=lambda x: (x['efficiency'], -x['gravityStrength']), reverse=True)
        
        tasks = []
        # Systematic modes
        for om in ['best', 'first']:
            for em in ['BFL', 'BLF', 'FBL']:
                for sm in ['vol', 'vol_w', 'height', 'footprint', 'weight', 'val_dens']:
                    tasks.append((sm, em, om))
        # Add random/perturbed variety
        for i in range(40):
            tasks.append(('p_' + str(i), random.choice(['BFL', 'BLF', 'FBL', 'FLB', 'LBF']), random.choice(['best', 'first'])))
        
        best_overall_cost = float('inf'); best_overall_sol = None
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_greedy_pack_worker, items, v_types, sm, em, om, i) for i, (sm, em, om) in enumerate(tasks)]
            for future in concurrent.futures.as_completed(futures):
                if time.time() - start_time > 550: break
                try:
                    sol_data, cost = future.result()
                    if cost < best_overall_cost:
                        best_overall_cost = cost; best_overall_sol = sol_data
                except Exception: pass
                    
        if best_overall_sol:
            self.sol = {'type_vehicle': [], 'idx_vehicle': [], 'id_item': [], 'x_origin': [], 'y_origin': [], 'z_origin': [], 'orient': []}
            for entry in best_overall_sol:
                for k in self.sol: self.sol[k].append(entry[k])
        self.write_solution_to_file()
