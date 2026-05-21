from instances import Instance
from solver_362540_362159_354745_360043.solver_362540_362159_354745_360043 import solver_362540_362159_354745_360043

if __name__ == '__main__':

    dataset_name = 'DatasetA'

    inst = Instance(dataset_name)

    solver = solver_362540_362159_354745_360043(inst)

    solver.solve()