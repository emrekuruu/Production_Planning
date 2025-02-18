import pyomo.environ as pyo
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import os

random.seed(42)
np.random.seed(42)

class Optimizer():
    def __init__(self, num_parts, num_colors, max_times, demand, parts_colors, alpha, 
                 unit_production_time, cleaning_time, machines, MIPGAP=0.01):
        """
        Parameters:
          - num_parts: integer number of parts (indexed 1,...,num_parts)
          - num_colors: number of colors (used in the objective via alpha)
          - max_times: list of maximum allowed processing times per machine
          - demand: dictionary mapping part in 1..num_parts (and dummy parts) to its demand.
                    (For dummy parts, demand should be 0.)
          - parts_colors: dict mapping part to a color (e.g., a matplotlib color string)
          - alpha: 2D list (indexed from 0) for the cleaning penalty between parts.
          - unit_production_time: production time per unit demand.
          - cleaning_time: cleaning time required when switching between parts of different colors.
          - machines: list of machine identifiers (assumed to be positive integers).
                      Dummy job for machine m is assumed to be numbered num_parts+m.
          - MIPGAP: relative MIP gap tolerance.
        """
        self.MIPGAP = MIPGAP
        self.num_parts = num_parts
        self.num_colors = num_colors
        self.max_times = max_times
        self.demand = demand
        self.parts_colors = parts_colors
        self.alpha = alpha
        self.unit_production_time = unit_production_time
        self.cleaning_time = cleaning_time
        self.machines = machines

        # Create a Pyomo ConcreteModel instance.
        self.model = pyo.ConcreteModel()
        
        # Calculate big M (an upper bound on time)
        total_processing_time = sum(self.demand[p] * self.unit_production_time 
                                    for p in range(1, self.num_parts + 1))
        total_cleaning_time = self.cleaning_time * (self.num_parts - 1)
        self.big_M = total_processing_time + total_cleaning_time

    def build_model(self):
        self._define_sets()
        self._define_parameters()
        self._define_variables()
        self._define_objective()
        self._define_constraints()
        
    def _define_sets(self):
        # Parts: 1..num_parts; dummy jobs: num_parts+1 ... num_parts+|machines|
        parts_plus_dummy = list(range(1, self.num_parts + len(self.machines) + 1))
        self.model.P = pyo.Set(initialize=range(1, self.num_parts + 1))   # real parts
        self.model.PD = pyo.Set(initialize=parts_plus_dummy)              # parts + dummy jobs
        self.model.M = pyo.Set(initialize=self.machines)                  # machines

    def _define_parameters(self):
        # Alpha parameter for parts: defined only on real parts (P x P)
        alpha_dict = {(p, q): self.alpha[p-1][q-1] 
                      for p in range(1, self.num_parts+1)
                      for q in range(1, self.num_parts+1)}
        self.model.alpha = pyo.Param(self.model.P, self.model.P, initialize=alpha_dict)

        # Filter out dummy jobs from the demand dictionary.
        real_demand = {p: self.demand[p] for p in range(1, self.num_parts+1)}
        self.model.demand = pyo.Param(self.model.P, initialize=real_demand)
        
        self.model.unit_time = pyo.Param(initialize=self.unit_production_time)
        self.model.cleaning_time = pyo.Param(initialize=self.cleaning_time)
        
        # Maximum times: if max_times is a list, map it to a dict keyed by machine identifier.
        if isinstance(self.max_times, list):
            max_times_dict = dict(zip(self.machines, self.max_times))
        else:
            max_times_dict = self.max_times
        self.model.max_times = pyo.Param(self.model.M, initialize=max_times_dict)
        
    def _define_variables(self):
        # Start times for both real parts and dummy jobs on each machine.
        self.model.start_times = pyo.Var(self.model.PD, self.model.M, domain=pyo.NonNegativeReals)
        # Predecessor binary variable: 1 if job p immediately precedes job q on machine m.
        self.model.predecessor = pyo.Var(self.model.PD, self.model.PD, self.model.M, domain=pyo.Binary)

    def _define_objective(self):
        # Objective: sum cleaning penalties only over transitions between real parts.
        def objective_rule(model):
            return sum(model.alpha[p, q] * model.predecessor[p, q, m]
                       for m in model.M
                       for p in model.P
                       for q in model.P)
        self.model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def _define_constraints(self):
        # 1. Dummy part constraints:
        # For each machine m, the dummy job (num_parts + m) must precede exactly one real part.
        def dummy_constraint_rule(model, m):
            dummy_job = self.num_parts + m
            return sum(model.predecessor[dummy_job, p, m] for p in model.P) == 1
        self.model.dummy_constr = pyo.Constraint(self.model.M, rule=dummy_constraint_rule)

        # 2. Single predecessor constraints:
        # Each real part must have exactly one predecessor (from parts or dummy) across all machines.
        def single_predecessor_rule(model, p):
            return sum(model.predecessor[q, p, m] 
                       for q in model.PD if q != p 
                       for m in model.M) == 1
        self.model.single_predecessor = pyo.Constraint(self.model.P, rule=single_predecessor_rule)

        # 3. Single successor constraints:
        # Each job (real part or dummy) may precede at most one real part over all machines.
        def single_successor_rule(model, p):
            return sum(model.predecessor[p, q, m]
                       for q in model.P if q != p
                       for m in model.M) <= 1
        self.model.single_successor = pyo.Constraint(self.model.PD, rule=single_successor_rule)

        # 4. No self-predecessor for real parts.
        def no_self_predecessor_rule(model, p, m):
            return model.predecessor[p, p, m] == 0
        self.model.no_self_predecessor = pyo.Constraint(self.model.P, self.model.M,
                                                         rule=no_self_predecessor_rule)

        # 5. Timing constraints:
        # For each job p (in PD) and each real part q on machine m:
        # If job p precedes part q, then p's finish time must not exceed q's start time.
        def timing_constraint_rule(model, p, q, m):
            if p == q:
                return pyo.Constraint.Skip
            # For real parts, processing time = demand[p]*unit_time; for dummy jobs, 0.
            if p <= self.num_parts:
                proc_time = model.demand[p] * model.unit_time
            else:
                proc_time = 0
            return model.start_times[p, m] <= (model.start_times[q, m] - proc_time +
                                               self.big_M * (1 - model.predecessor[p, q, m]))
        self.model.timing_constr = pyo.Constraint(self.model.PD, self.model.P, self.model.M,
                                                   rule=timing_constraint_rule)

        # 6. Machine time limits:
        # For each machine m and each real part p, the finish time must not exceed m's max time.
        def machine_time_rule(model, m, p):
            return (model.start_times[p, m] + model.demand[p] * model.unit_time
                    <= model.max_times[m] + self.big_M * (1 - sum(model.predecessor[q, p, m] 
                                                                  for q in model.PD))
                   )
        self.model.machine_time_constr = pyo.Constraint(self.model.M, self.model.P,
                                                          rule=machine_time_rule)

        # 7. Cleaning time constraints:
        # For each pair of distinct real parts p and q on machine m, if they have different colors,
        # then cleaning time is added after p.
        def cleaning_time_rule(model, p, q, m):
            if p == q or self.parts_colors[p] == self.parts_colors[q]:
                return pyo.Constraint.Skip
            return (model.start_times[p, m] + model.demand[p] * model.unit_time 
                    + model.cleaning_time
                    <= model.start_times[q, m] + self.big_M * (1 - model.predecessor[p, q, m])
                   )
        self.model.cleaning_constr = pyo.Constraint(self.model.P, self.model.P, self.model.M,
                                                     rule=cleaning_time_rule)

        # 8. Consistency constraints:
        # For each pair of distinct real parts p and q on machine m,
        # if part p is chosen to precede part q then p must have a predecessor.
        def consistency_constraint_rule(model, p, q, m):
            if p == q:
                return pyo.Constraint.Skip
            return sum(model.predecessor[r, p, m] for r in model.PD) >= model.predecessor[p, q, m]
        self.model.consistency_constr = pyo.Constraint(self.model.P, self.model.P, self.model.M,
                                                        rule=consistency_constraint_rule)

    def __call__(self, solver_name='cbc', verbose=True):
        """
        Build and solve the model.
        
        Parameters:
          - solver_name: name of the solver to use (default 'cbc' for open-source)
          - verbose: if True, display solver output.
          
        Returns:
          - results: solver results object
          - optimization time in seconds
        """
        self.build_model()
        solver = pyo.SolverFactory(solver_name)
        if not verbose:
            solver.options['tee'] = False
        solver.options['MIPGap'] = self.MIPGAP
        solver.options['threads'] = 8
        solver.options['Seed'] = 12345

        start_time = time.time()
        results = solver.solve(self.model, tee=verbose)
        end_time = time.time()

        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError("Optimization failed!")
            
        return results, end_time - start_time

    def visualize(self):
        """
        Visualize the schedule as a Gantt chart.
        """
        # Extract start times for real parts (ignore dummy jobs)
        start_times_values = {
            (p, m): pyo.value(self.model.start_times[p, m])
            for p in self.model.P for m in self.model.M
        }
        # Compute end times: finish = start time + processing time
        end_times_values = {
            (p, m): (pyo.value(self.model.start_times[p, m]) + self.demand[p] * self.unit_production_time)
            if sum(pyo.value(self.model.predecessor[q, p, m]) for q in self.model.PD) == 1 else 0
            for p in self.model.P for m in self.model.M
        }

        # Placeholder for breaks (adjust if needed)
        breaks = {m: [(0, 0, 0)] for m in self.machines}

        num_machines = len(self.machines)
        fig, axs = plt.subplots(num_machines, 1, figsize=(22, 6 * num_machines), sharex=True)
        if num_machines == 1:
            axs = [axs]

        # Determine global max time (if max_times is a list or dict)
        if isinstance(self.max_times, list):
            global_max_time = max(self.max_times)
        else:
            global_max_time = max(self.max_times[m] for m in self.machines)

        for i, m in enumerate(self.machines):
            ax = axs[i]
            cumulative_shift = 0  
            # Sort parts by their start time on machine m.
            sorted_parts = sorted(list(self.model.P), key=lambda p: start_times_values[(p, m)])
            for p in sorted_parts:
                part_color = self.parts_colors[p]
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
                ax.text((start + end) / 2, p, f'Type {p}', ha='center', va='center', color='white')
            # Draw max time vertical line
            if isinstance(self.max_times, list):
                m_max = self.max_times[m-1]
            else:
                m_max = self.max_times[m]
            ax.axvline(x=m_max, color='red', linestyle='--', alpha=0.8,
                       linewidth=1.5, label=f"Max Time Machine {m}")
            ax.legend(loc='upper right')
            ax.set_title(f"Machine {m} Schedule", fontsize=16)
            ax.set_ylabel("Product Type", fontsize=14)
            ax.set_yticks(list(self.model.P))
            ax.set_yticklabels([f'Type {i}' for i in self.model.P])
            ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.xlabel("Time (seconds)", fontsize=14)
        plt.xticks(np.arange(0, global_max_time, 4000))
        fig.suptitle("Optimal Production Schedule for Machines", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
