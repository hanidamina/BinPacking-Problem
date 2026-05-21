import os
import pandas as pd
from instances import Instance

datasets = [
    'Dataset0', 'Dataset1', 'Dataset2', 'Dataset3', 'Dataset4',
    'Dataset5', 'Dataset6', 'Dataset7', 'Dataset8', 'Dataset9',
    'DatasetA', 'DatasetB', 'DatasetC', 'DatasetD', 'DatasetE',
    'DatasetF', 'DatasetG', 'DatasetH', 'DatasetI', 'DatasetJ'
]

solver_name = 'solver_362540_362159_354745_360043'

def get_dims(item, orient):
    w, d, h = item["width"], item["depth"], item["height"]
    rotations = [
        (w, d, h),
        (d, w, h),
        (h, d, w),
        (d, h, w),
        (w, h, d),
        (h, w, d),
    ]
    return rotations[orient]

def overlap_1d(a_min, a_max, b_min, b_max):
    return max(a_min, b_min) < min(a_max, b_max)

def boxes_overlap(a, b):
    return (
        overlap_1d(a["x1"], a["x2"], b["x1"], b["x2"]) and
        overlap_1d(a["y1"], a["y2"], b["y1"], b["y2"]) and
        overlap_1d(a["z1"], a["z2"], b["z1"], b["z2"])
    )

if __name__ == '__main__':
    all_feasible = True
    total_total_cost = 0
    for ds in datasets:
        print(f"Checking {ds}...", end=" ")
        inst = Instance(ds)
        items = inst.df_items
        vehicles = inst.df_vehicles
        sol_path = os.path.join('results', f'sol_{ds}_{solver_name}.csv')
        if not os.path.exists(sol_path):
            print(f"FAILED: {sol_path} not found")
            all_feasible = False
            continue
        
        solution = pd.read_csv(sol_path)
        items_dict = items.to_dict("index")
        vehicles_dict = vehicles.to_dict("index")
        groups = solution.groupby("idx_vehicle")

        total_cost = 0
        feasible = True
        placed_item_ids = set()

        for vidx, group in groups:
            vehicle_type = group.iloc[0]['type_vehicle']
            vehicle = vehicles_dict[vehicle_type]
            placed_boxes = []
            total_weight = 0
            total_value = 0
            total_cost += vehicle["cost"]

            for _, row in group.iterrows():
                item_id = row["id_item"]
                placed_item_ids.add(item_id)
                item = items_dict[item_id]
                orient = int(row["orient"])
                w, d, h = get_dims(item, orient)
                x, y, z = row["x_origin"], row["y_origin"], row["z_origin"]
                box = {
                    "id": item_id,
                    "x1": x, "y1": y, "z1": z,
                    "x2": x + d, "y2": y + w, "z2": z + h,
                    "base_area": w * d
                }

                if box["x2"] > vehicle["depth"] + 1e-6 or \
                   box["y2"] > vehicle["width"] + 1e-6 or \
                   box["z2"] > vehicle["height"] + 1e-6:
                    feasible = False
                for other in placed_boxes:
                    if boxes_overlap(box, other):
                        feasible = False
                placed_boxes.append(box)
                total_weight += item["weight"]
                total_value += item["value"]

            if total_weight > vehicle["maxWeight"] + 1e-6:
                feasible = False
            if not pd.isna(vehicle["maxValue"]) and total_value > vehicle["maxValue"] + 1e-6:
                feasible = False

            gravity = vehicle["gravityStrength"]
            for i, box in enumerate(placed_boxes):
                if box["z1"] == 0: continue
                support_area = 0
                for j, other in enumerate(placed_boxes):
                    if i == j: continue
                    if abs(other["z2"] - box["z1"]) < 1e-6:
                        dx = max(0, min(box["x2"], other["x2"]) - max(box["x1"], other["x1"]))
                        dy = max(0, min(box["y2"], other["y2"]) - max(box["y1"], other["y1"]))
                        support_area += dx * dy
                if support_area < box["base_area"] * (gravity / 100.0) - 1e-9:
                    feasible = False

        if len(placed_item_ids) != len(items):
            feasible = False
            
        if feasible:
            print(f"OK. Cost: {total_cost:.2f}")
            total_total_cost += total_cost
        else:
            print("FAILED.")
            all_feasible = False

    print("---------------------------------")
    if all_feasible:
        print(f"ALL SOLUTIONS ARE FEASIBLE. Total global cost: {total_total_cost:.2f}")
    else:
        print("SOME SOLUTIONS ARE INFEASIBLE.")
