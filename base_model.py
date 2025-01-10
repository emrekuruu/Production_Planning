import gurobipy as gp
from gurobipy import GRB, quicksum
import random
import os
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, num_parts=10, group_length=2, max_time_machine_A=28000, max_time_machine_B=19000):
        self.num_parts = num_parts
        self.group_length = group_length
        self.max_time_machine_A = max_time_machine_A
        self.max_time_machine_B = max_time_machine_B
        self.unit_production_time = 80
        self.cleaning_time = 80
        self.demand = {p: 50 for p in range(1, num_parts + 1)}
        self.demand[num_parts + 1] = 0
        self.demand[num_parts + 2] = 0
        self.color_cost = {
            'Red': 10, 'Blue': 10, 'Green': 10, 'Yellow': 10,
            'Black': 10, 'Pink': 10, 'Purple': 10,
        }
        self.parts_colors = self._assign_colors()
        self.alpha = self._compute_alpha()
        self.machines = [1, 2]

    def _assign_colors(self):
        parts_colors = {}
        for i in range(1, self.num_parts + 1, self.group_length):
            color = random.choice(list(self.color_cost.keys()))
            while color in parts_colors.values():
                color = random.choice(list(self.color_cost.keys()))
            for j in range(self.group_length):
                parts_colors[i + j] = color
        return parts_colors

    def _compute_alpha(self):
        return [[abs(1 - (self.parts_colors[p] == self.parts_colors[q])) 
                 for q in range(1, self.num_parts + 1)] 
                for p in range(1, self.num_parts + 1)]

    def create_model(self):
        model = gp.Model("PaintingProcessOptimization")

        # Variables
        self.start_times = model.addVars(range(1, self.num_parts + 3), self.machines, 
                                    vtype=GRB.CONTINUOUS, name="self.start_times")
        self.predecessor = model.addVars(range(1, self.num_parts + 3), range(1, self.num_parts + 3), 
                                     self.machines, vtype=GRB.BINARY, name="self.predecessor")

        # Objective
        model.setObjective(
            quicksum(self.alpha[p-1][q-1] * self.predecessor[p, q, m]
                     for p in range(1, self.num_parts + 1)
                     for q in range(1, self.num_parts + 1)
                     for m in self.machines),
            GRB.MINIMIZE
        )

        # Constraints
        big_M = self.max_time_machine_A

        # Dummy part constraints
        model.addConstr(quicksum(self.predecessor[12, p, 1] for p in range(1, self.num_parts + 1)) == 1, 
                        name="Dummy_12_Predecessor_Machine_1")
        model.addConstr(quicksum(self.predecessor[11, p, 2] for p in range(1, self.num_parts + 1)) == 1, 
                        name="Dummy_11_Predecessor_Machine_2")

        # self.predecessor constraints
        for p in range(1, self.num_parts + 1):
            model.addConstr(
                quicksum(self.predecessor[q, p, m] for q in range(1, self.num_parts + 3) for m in self.machines if q != p) == 1,
                name=f"Single_predecessor_{p}"
            )
        for p in range(1, self.num_parts + 3):
            model.addConstr(
                quicksum(self.predecessor[p, q, m] for q in range(1, self.num_parts + 1) for m in self.machines if q != p) <= 1,
                name=f"Single_successor_{p}"
            )
        for p in range(1, self.num_parts + 1):
            for m in self.machines:
                model.addConstr(self.predecessor[p, p, m] == 0, name=f"No_self_predecessor_{p}_{m}")

        # Consistency constraints
        for p in range(1, self.num_parts + 1):
            for q in range(1, self.num_parts + 1):
                if p != q:
                    for m in self.machines:
                        model.addConstr(
                            quicksum(self.predecessor[r, p, m] for r in range(1, self.num_parts + 3)) >= self.predecessor[p, q, m],
                            name=f"Predecessor_Consistency_{p}_{q}_Machine_{m}"
                        )

        # Timing constraints
        for p in range(1, self.num_parts + 3):
            for q in range(1, self.num_parts + 1):
                if p != q:
                    for m in self.machines:
                        model.addConstr(
                            self.start_times[p, m] <= (self.start_times[q, m] - self.demand[p] * self.unit_production_time) + 
                            big_M * (1 - self.predecessor[p, q, m]),
                            name=f"No_Overlap_Same_Machine_{p}_{q}_Machine_{m}"
                        )

        # Machine time limits
        for p in range(1, self.num_parts + 1):
            model.addConstr(
                self.start_times[p, 1] + self.demand[p] * self.unit_production_time <= 
                self.max_time_machine_A + big_M * (1 - quicksum(self.predecessor[q, p, 1] for q in range(1, self.num_parts + 3))),
                name=f"Max_Time_Machine_A_{p}"
            )
            model.addConstr(
                self.start_times[p, 2] + self.demand[p] * self.unit_production_time <= 
                self.max_time_machine_B + big_M * (1 - quicksum(self.predecessor[q, p, 2] for q in range(1, self.num_parts + 3))),
                name=f"Max_Time_Machine_B_{p}"
            )

        # Cleaning time constraints
        for p in range(1, self.num_parts + 1):
            for q in range(1, self.num_parts + 1):
                if p != q and self.parts_colors[p] != self.parts_colors[q]:
                    for m in self.machines:
                        model.addConstr(
                            self.start_times[p, m] + self.demand[p] * self.unit_production_time + self.cleaning_time <= 
                            self.start_times[q, m] + big_M * (1 - self.predecessor[p, q, m]),
                            name=f"Cleaning_Time_{p}_{q}_Machine_{m}"
                        )

        return model

    def __call__(self):
        model = self.create_model()
        model.setParam("IntFeasTol", 1e-9)
        model.setParam("Threads", 8) 
        self.optimize(model)
        return model
    
    @abstractmethod
    def optimize(self, model):
        pass


class No_Heuristic(BaseModel):
    def optimize(self, model):
        model.optimize()
        return model