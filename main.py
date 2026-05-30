from instances import Instance
from solver_362540 import solver_362540

if __name__ == '__main__':

    dataset_name = 'DatasetE'

    inst = Instance(dataset_name)

    solver = solver_362540(inst)

    solver.solve()