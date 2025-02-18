import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import (ConcreteModel, Var, Param, Set, RangeSet, Binary,
                           NonNegativeReals, Constraint, Objective, SolverFactory,
                           summation, value, ConstraintList)
                           
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class Optimizer:
    def __init__(self, num_parts, num_colors, max_times, demand, parts_colors,
                 alpha, unit_production_time, cleaning_time, machines, MIPGAP=0.01):
        """
        Parameters:
          num_parts: integer number of parts (jobs) (indexed 1,...,num_parts)
          num_colors: number of colors (used in objective via alpha)
          max_times: list (or dict) of maximum allowed processing times per machine;
          demand: dict mapping part in 1..num_parts to its demand (production quantity)
          parts_colors: dict mapping part (1..num_parts) to its color (e.g., a color string)
          alpha: precomputed matrix (list of lists) for the cleaning penalty between parts.
                 (For instance, alpha[p-1][q-1] = abs(1 - (parts_colors[p]==parts_colors[q]))
          unit_production_time: production time per unit demand.
          cleaning_time: cleaning time required between jobs with different colors.
          machines: list of machine identifiers. (For dummy jobs we assume machines are numbered
                    with integers starting at 1; dummy index for machine m is num_parts+m.)
          MIPGAP: not used in CBC but kept for compatibility.
        """
        self.num_parts = num_parts
        self.num_colors = num_colors
        self.max_times = max_times  # assume a dict or list with machine order corresponding to machines
        self.demand = demand        # dictionary: key in {1,...,num_parts}
        self.parts_colors = parts_colors  # dictionary: key in {1,...,num_parts}
        self.alpha = alpha          # 2D list: indices [p-1][q-1] for p,q in parts (only)
        self.unit_production_time = unit_production_time
        self.cleaning_time = cleaning_time
        self.machines = machines    # list of machine identifiers (assumed to be integers)
        self.MIPGAP = MIPGAP

        # Create an internal structure for the dummy job indices.
        # For each machine m in machines, we assign a dummy job with index: num_parts + m.
        self.dummy = {m: self.num_parts + m for m in self.machines}

    def create_model(self):
        model = ConcreteModel(name="PaintingProcessOptimization")

        # Define sets:
        model.PARTS = RangeSet(1, self.num_parts)
        # Dummy jobs: one per machine.
        model.DUMMY = Set(initialize=[self.dummy[m] for m in self.machines])
        # Jobs = parts ∪ dummy jobs.
        model.JOBS = Set(initialize=list(range(1, self.num_parts+1)) + [self.dummy[m] for m in self.machines])
        model.MACHINES = Set(initialize=self.machines)

        # Parameters
        # Production demand: for parts use given demand; for dummy jobs, set demand to 0.
        def demand_init(model, j):
            if j in model.PARTS:
                return self.demand[j]
            else:
                return 0
        model.demand = Param(model.JOBS, initialize=demand_init)

        # Maximum allowed time per machine. Assume self.max_times is a dict keyed by machine.
        model.max_time = Param(model.MACHINES, initialize=lambda model, m: self.max_times[m])

        model.unit_production_time = Param(initialize=self.unit_production_time)
        model.cleaning_time = Param(initialize=self.cleaning_time)

        # For parts (only) we provide the cleaning penalty alpha. (Only defined for PARTS x PARTS.)
        def alpha_init(model, p, q):
            # p and q are parts: they are defined only on PARTS.
            return self.alpha[p-1][q-1]
        model.alpha = Param(model.PARTS, model.PARTS, initialize=alpha_init)

        # Compute big_M constant.
        total_processing_time = sum(self.demand[p] * self.unit_production_time for p in range(1, self.num_parts+1))
        total_cleaning_time = self.cleaning_time * (self.num_parts - 1)
        big_M = total_processing_time + total_cleaning_time
        model.big_M = Param(initialize=big_M)

        # Decision variables:
        # start_times for every job (parts and dummy) on every machine.
        model.start_time = Var(model.JOBS, model.MACHINES, domain=NonNegativeReals)
        # predecessor: binary variable: predecessor[p,q,m] = 1 if job p immediately precedes job q on machine m.
        model.predecessor = Var(model.JOBS, model.JOBS, model.MACHINES, domain=Binary)

        # Objective:
        # Sum only over parts-to-parts transitions (dummy jobs are not in objective).
        def obj_rule(model):
            return sum(model.alpha[p, q] * model.predecessor[p, q, m]
                       for m in model.MACHINES
                       for p in model.PARTS
                       for q in model.PARTS)
        model.OBJ = Objective(rule=obj_rule, sense=1)  # minimize

        # Constraints:
        model.constraints = ConstraintList()

        # 1. Dummy part constraints: For each machine m, the dummy job must precede exactly one part.
        for m in model.MACHINES:
            dummy_j = self.dummy[m]
            model.constraints.add(
                sum(model.predecessor[dummy_j, p, m] for p in model.PARTS) == 1
            )

        # 2. For each part p, there is exactly one predecessor (from any job in JOBS, excluding p itself) over all machines.
        for p in model.PARTS:
            model.constraints.add(
                sum(model.predecessor[q, p, m] for m in model.MACHINES for q in model.JOBS if q != p) == 1
            )

        # 3. For each job p (in JOBS), there is at most one successor (from PARTS only) over all machines.
        for p in model.JOBS:
            model.constraints.add(
                sum(model.predecessor[p, q, m] for m in model.MACHINES for q in model.PARTS if q != p) <= 1
            )

        # 4. No self-predecessor for parts on any machine.
        for p in model.PARTS:
            for m in model.MACHINES:
                model.constraints.add(
                    model.predecessor[p, p, m] == 0
                )

        # 5. Consistency constraints:
        #    For each part p and each part q (with p != q) and machine m,
        #    if job p precedes job q then job p must have a predecessor.
        for p in model.PARTS:
            for q in model.PARTS:
                if p != q:
                    for m in model.MACHINES:
                        model.constraints.add(
                            sum(model.predecessor[r, p, m] for r in model.JOBS) >= model.predecessor[p, q, m]
                        )

        # 6. Timing constraints: if p precedes q on machine m then
        #    start_time[p, m] + processing time of p <= start_time[q, m] plus big_M slack.
        for m in model.MACHINES:
            for p in model.JOBS:
                for q in model.PARTS:
                    if p != q:
                        proc_time_p = model.demand[p] * model.unit_production_time
                        model.constraints.add(
                            model.start_time[p, m] <= model.start_time[q, m] - proc_time_p
                            + model.big_M * (1 - model.predecessor[p, q, m])
                        )

        # 7. Machine time limits:
        # For each machine m and each part p, if p is scheduled then its finish time must not exceed the machine’s max_time.
        for m in model.MACHINES:
            for p in model.PARTS:
                proc_time_p = model.demand[p] * model.unit_production_time
                model.constraints.add(
                    model.start_time[p, m] + proc_time_p
                    <= model.max_time[m] + model.big_M * (1 - sum(model.predecessor[q, p, m] for q in model.JOBS))
                )

        # 8. Cleaning time constraints:
        # For each pair of parts p and q (with different colors) and on each machine, if p precedes q then add cleaning time.
        for m in model.MACHINES:
            for p in model.PARTS:
                for q in model.PARTS:
                    if p != q and (self.parts_colors[p] != self.parts_colors[q]):
                        proc_time_p = model.demand[p] * model.unit_production_time
                        model.constraints.add(
                            model.start_time[p, m] + proc_time_p + model.cleaning_time
                            <= model.start_time[q, m] + model.big_M * (1 - model.predecessor[p, q, m])
                        )

        self.model = model  # store model in the instance (for later use)
        return model

    def _set_initial_solution(self, initial_solution):
        """
        initial_solution: dictionary with keys = machine (from self.machines) and value = list of tuples
           (part, start_time, end_time).
        """
        # Set initial values for start_time and predecessor variables.
        model = self.model

        # Set all start times initially to 0.
        for j in model.JOBS:
            for m in model.MACHINES:
                model.start_time[j, m].value = 0

        for p in model.JOBS:
            for q in model.JOBS:
                for m in model.MACHINES:
                    model.predecessor[p, q, m].value = 0

        # For each machine, set the start time of the scheduled parts and the ordering
        for m, job_list in initial_solution.items():
            if not job_list:
                continue
            # sort by start time
            job_list_sorted = sorted(job_list, key=lambda x: x[1])
            for (part, start_time, end_time) in job_list_sorted:
                # set the start time for that part on machine m
                if part in self.model.PARTS:
                    model.start_time[part, m].value = start_time
            # For consecutive jobs, set the predecessor variable.
            for i in range(len(job_list_sorted) - 1):
                prev_part = job_list_sorted[i][0]
                next_part = job_list_sorted[i+1][0]
                model.predecessor[prev_part, next_part, m].value = 1
            # Set dummy predecessor: dummy job for machine m precedes the first job.
            dummy_j = self.dummy[m]
            first_job = job_list_sorted[0][0]
            model.predecessor[dummy_j, first_job, m].value = 1

    def __call__(self, initial_solution=None, verbose=True):
        model = self.create_model()

        if initial_solution is not None:
            self._set_initial_solution(initial_solution)

        # Create solver (using CBC)
        solver = SolverFactory('cbc')
        # You can set solver options if desired, e.g., time limits etc.
        # For example: solver.options['seconds'] = 3600

        start_time = time.time()
        result = solver.solve(model, tee=verbose)
        end_time = time.time()
        optimization_time = end_time - start_time

        # Check solver termination status
        if (result.solver.termination_condition).name == "infeasible":
            raise Exception("Model is infeasible. Please check the constraints and data.")

        if verbose:
            # In Pyomo there is no direct attribute for number of quadratic terms, so we just print termination info.
            print("Solver Status:", result.solver.status)
            print("Termination Condition:", result.solver.termination_condition)
            print("Optimization Time: {:.2f} seconds".format(optimization_time))

        obj_val = value(model.OBJ)
        return model, obj_val, optimization_time

    def visualize(self):
        """
        Create a Gantt chart-like schedule.
        (This function assumes that after solving, self.model is available and variable values are set.)
        """
        model = self.model
        # Extract start times for parts only (ignore dummy jobs)
        start_times_values = {(p, m): value(model.start_time[p, m])
                              for p in model.PARTS for m in model.MACHINES}

        end_times_values = {}
        for p in model.PARTS:
            for m in model.MACHINES:
                proc_time = self.demand[p] * self.unit_production_time
                # Only consider parts that are scheduled (check if any predecessor sum equals 1)
                pred_sum = sum(value(model.predecessor[q, p, m]) for q in model.JOBS)
                if pred_sum is None or pred_sum < 0.5:
                    end_times_values[(p, m)] = 0
                else:
                    end_times_values[(p, m)] = start_times_values[(p, m)] + proc_time

        # Placeholder for breaks; adjust as needed.
        breaks = {m: [(0, 0, 0)] for m in self.machines}

        num_machines = len(self.machines)
        fig, axs = plt.subplots(num_machines, 1, figsize=(22, 6 * num_machines), sharex=True)
        if num_machines == 1:
            axs = [axs]

        global_max_time = max(self.max_times[m] for m in self.machines)

        for i, m in enumerate(self.machines):
            ax = axs[i]
            cumulative_shift = 0
            # Sort parts by their start time on machine m.
            sorted_parts = sorted([p for p in model.PARTS], key=lambda p: start_times_values[(p, m)])
            for p in sorted_parts:
                part_color = self.parts_colors[p]  # expect a color string
                start = start_times_values[(p, m)]
                end = end_times_values[(p, m)]
                if end == 0:
                    continue
                shifted_start = start + cumulative_shift
                shifted_end = end + cumulative_shift

                # Check if any break overlaps (this logic is illustrative)
                for (break_start, break_end, shift_amount) in breaks[m]:
                    if shifted_start < break_start < shifted_end:
                        ax.barh(p, break_start - shifted_start, left=shifted_start,
                                color=part_color, edgecolor='black')
                        shifted_start = break_end + shift_amount
                        shifted_end += shift_amount
                        ax.barh(p, end - break_end, left=shifted_start,
                                color=part_color, edgecolor='black')
                        cumulative_shift += shift_amount
                        break
                    elif shifted_start >= break_end:
                        shifted_start += shift_amount
                        shifted_end += shift_amount

                ax.barh(p, shifted_end - shifted_start, left=shifted_start,
                        color=part_color, edgecolor='black')
                ax.text((shifted_start + shifted_end) / 2, p, f'Part {p}', ha='center',
                        va='center', color='white')

            ax.axvline(x=self.max_times[m], color='red', linestyle='--',
                       alpha=0.8, linewidth=1.5, label=f"Max Time Machine {m}")
            ax.legend(loc='upper right')
            ax.set_title(f"Machine {m} Schedule", fontsize=16)
            ax.set_ylabel("Part", fontsize=14)
            ax.set_yticks(list(model.PARTS))
            ax.set_yticklabels([f'Part {i}' for i in model.PARTS])
            ax.grid(axis='x', linestyle='--', alpha=0.7)

        plt.xlabel("Time", fontsize=14)
        plt.xticks(np.arange(0, global_max_time + 1000, 4000))
        fig.suptitle("Optimal Production Schedule for Machines", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

