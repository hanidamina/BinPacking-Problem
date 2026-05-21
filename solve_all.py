import os
from instances import Instance
from solver_362540_362159_354745_360043.solver_362540_362159_354745_360043 import solver_362540_362159_354745_360043

datasets = [
    'Dataset0', 'Dataset1', 'Dataset2', 'Dataset3', 'Dataset4',
    'Dataset5', 'Dataset6', 'Dataset7', 'Dataset8', 'Dataset9',
    'DatasetA', 'DatasetB', 'DatasetC', 'DatasetD', 'DatasetE',
    'DatasetF', 'DatasetG', 'DatasetH', 'DatasetI', 'DatasetJ'
]

if __name__ == '__main__':
    for ds in datasets:
        print(f"Solving {ds}...")
        inst = Instance(ds)
        solver = solver_362540_362159_354745_360043(inst)
        solver.solve()
        print(f"Finished {ds}.")
