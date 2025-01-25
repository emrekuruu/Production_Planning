import gurobipy as gp
from gurobipy import GRB, quicksum
import random
import time 
import random
import matplotlib.pyplot as plt
import numpy as np
import os 

os.environ["GRB_LICENSE_FILE"] = "/Users/emrekuru/Developer/Production_Planning/gurobi.lic"

class Optimizer():
    def __init__(self, num_parts, group_length, max_time_machine_A, max_time_machine_B, demand, parts_colors, alpha, unit_production_time, cleaning_time):
        self.num_parts = num_parts
        self.group_length = group_length
        self.max_time_machine_A = max_time_machine_A
        self.max_time_machine_B = max_time_machine_B
        self.demand = demand
        self.parts_colors = parts_colors
        self.alpha = alpha
        self.unit_production_time = unit_production_time
        self.cleaning_time = cleaning_time
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
        model.addConstr(quicksum(self.predecessor[len(self.parts_colors)+1, p, 1] for p in range(1, self.num_parts + 1)) == 1, 
                        name="Dummy_Predecessor_Machine_1")
        model.addConstr(quicksum(self.predecessor[len(self.parts_colors)+2, p, 2] for p in range(1, self.num_parts + 1)) == 1, 
                        name="Dummy_Predecessor_Machine_2")

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

    def _set_initial_solution(self, initial_solution):
        # 1) Reset all starts to 0 (optional, if not done elsewhere)
        for (p, m) in self.start_times.keys():
            self.start_times[p, m].Start = 0
        for (pA, pB, m) in self.predecessor.keys():
            self.predecessor[pA, pB, m].Start = 0

        # 2) Fill in the user-given schedule
        for m, job_list in initial_solution.items():
            job_list_sorted = sorted(job_list, key=lambda x: x[1])
            for (part, start_time, end_time) in job_list_sorted:
                self.start_times[part, m].Start = start_time

            for i in range(len(job_list_sorted) - 1):
                prev_part = job_list_sorted[i][0]
                next_part = job_list_sorted[i+1][0]
                self.predecessor[prev_part, next_part, m].Start = 1

        # 3) Now assign dummy → first job for each machine
        dummy_1 = len(self.parts_colors) + 1  # Dummy for Machine 1
        dummy_2 = len(self.parts_colors) + 2  # Dummy for Machine 2

        # Machine 1
        if 1 in initial_solution and initial_solution[1]:
            first_job = sorted(initial_solution[1], key=lambda x: x[1])[0]
            self.predecessor[dummy_1, first_job[0], 1].Start = 1

        # Machine 2
        if 2 in initial_solution and initial_solution[2]:
            first_job = sorted(initial_solution[2], key=lambda x: x[1])[0]
            self.predecessor[dummy_2, first_job[0], 2].Start = 1


    def __call__(self, initial_solution=None):
        model = self.create_model()
        model.setParam("IntFeasTol", 1e-9)
        model.setParam("Threads", 8)

        if initial_solution is not None:
            self._set_initial_solution(initial_solution)

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        optimization_time = end_time - start_time

        # Check the number of quadratic constraints and non-zero quadratic terms in the objective
        num_quadratic_constraints = model.getAttr("NumQConstrs")  # Number of quadratic constraints
        num_quadratic_obj_terms = model.getAttr("NumQNZs")       # Number of non-zero quadratic terms in the objective

        # Print the results to confirm linearity
        print("Number of quadratic constraints:", num_quadratic_constraints)
        print("Number of quadratic objective terms:", num_quadratic_obj_terms)

        # Check if the model is MILP based on the absence of quadratic terms
        is_milp = num_quadratic_constraints == 0 and num_quadratic_obj_terms == 0 and model.getAttr("IsMIP")
        print("Model is MILP:", bool(is_milp))

        # Check if the model is infeasible
        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write("model.ilp")

            # Print the constraints that are part of the IIS
            print("\nThe following constraints are part of the IIS:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"{c.constrName}")

            raise Exception("Model is infeasible. Check the IIS output for more details.")

        return model, optimization_time
    
    def optimize(self, model):
        model.optimize()
        return model
    
    def visualize(self):

        start_times_values = {
            (p, m): self.start_times[p, m].X
            for p in range(1, self.num_parts + 1)
            for m in self.machines
        }

        end_times_values = {
            (p, m): (self.start_times[p, m].X + self.demand[p] * self.unit_production_time)
            if sum(self.predecessor[q, p, m].X for q in range(1, self.num_parts + 3)) == 1 else 0
            for p in range(1, self.num_parts + 1)
            for m in self.machines
        }

        breaks = {
            1: [(0, 0, 0)],  # Breaks for Machine 1
            2: [(0, 0, 0)],  # Breaks for Machine 2
        }

        fig, axs = plt.subplots(2, 1, figsize=(22, 12), sharex=True)

        color_map = {
            'Red': 'red',
            'Blue': 'blue',
            'Green': 'green',
            'Yellow': 'yellow',
            'Black': 'black',
            'Purple': 'purple',
            'Orange': 'orange',
            'Pink': 'pink',
        }

        max_time_A = self.max_time_machine_A
        max_time_B = self.max_time_machine_B

        for m in self.machines:
            ax = axs[m - 1] 
            cumulative_shift = 0  #

            sorted_parts = sorted(
                [p for p in range(1, self.num_parts + 1)],
                key=lambda p: start_times_values[(p, m)]
            )

            for p in sorted_parts:
                part_color = color_map[self.parts_colors[p]]
                start = start_times_values[(p, m)]
                end = end_times_values[(p, m)]

                if end == 0:
                    continue

                shifted_start = start + cumulative_shift
                shifted_end = end + cumulative_shift

                if shifted_start == shifted_end:
                    continue

                for break_start, break_end, shift_amount in breaks[m]:
                    if shifted_start < break_start < shifted_end:
                        ax.barh(p, break_start - shifted_start, left=shifted_start, color=part_color, edgecolor='black')

                        shifted_start = break_end + shift_amount
                        shifted_end += shift_amount
                        ax.barh(p, end - break_end, left=shifted_start, color=part_color, edgecolor='black')

                        cumulative_shift += shift_amount
                        break 
                    elif shifted_start >= break_end:
                        shifted_start += shift_amount
                        shifted_end += shift_amount

                ax.barh(p, shifted_end - shifted_start, left=shifted_start, color=part_color, edgecolor='black')
                ax.text((start + end) / 2, p, f'Type {p}', ha='center', va='center', color='white')

            if m == 1:
                ax.axvline(x=max_time_A, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label="Max Time A")
                ax.legend(loc='upper right')
            elif m == 2:
                ax.axvline(x=max_time_B, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label="Max Time B")
                ax.legend(loc='upper right')

            ax.set_title(f"Machine {m} Schedule", fontsize=16)
            ax.set_ylabel("Product Type", fontsize=14)
            ax.set_yticks(range(1, self.num_parts + 1))
            ax.set_yticklabels([f'Type {i}' for i in range(1, self.num_parts + 1)])
            ax.grid(axis='x', linestyle='--', alpha=0.7)

        plt.xlabel("Time (seconds)", fontsize=14)
        plt.xticks(np.arange(0, max_time_A, 4000))  

        fig.suptitle("Optimal Production Schedule for Two Machines", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

        plt.show()

