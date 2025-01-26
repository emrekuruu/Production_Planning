import random
import time
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import time
import numpy as np

random.seed(42)
np.random.seed(42)

class BaseScheduler(ABC):
    def __init__(self, num_parts, group_length, max_time_machine_A, max_time_machine_B,
                 demand, parts_colors, alpha, unit_production_time, 
                 cleaning_time, machines):
        self.num_parts = num_parts
        self.group_length = group_length
        self.max_times = {1: max_time_machine_A, 2: max_time_machine_B}
        self.demand = demand
        self.parts_colors = parts_colors
        self.alpha = alpha
        self.unit_production_time = unit_production_time
        self.cleaning_time = cleaning_time
        self.machines = machines
        self.type = None
        
        self.schedule = {1: [], 2: []}
        self.objective_value = 0

    def calculate_processing_time(self, part):
        return self.demand[part] * self.unit_production_time

    def add_cleaning_times(self, schedule):
        cleaned = {1: [], 2: []}
        for machine in self.machines:
            jobs = sorted(schedule[machine], key=lambda x: x[1])
            t = 0
            prev_color = None
            for (part, _, _) in jobs:
                color = self.parts_colors[part]
                if prev_color and color != prev_color:
                    t += self.cleaning_time
                p_time = self.calculate_processing_time(part)
                start = t
                end = start + p_time

                cleaned[machine].append((part, start, end))
                t = end
                prev_color = color
        return cleaned

    def validate_schedule(self, schedule):

        cleaned = self.add_cleaning_times(schedule)
        for machine in self.machines:
            current_time = 0
            prev_color = None
            for job in sorted(cleaned[machine], key=lambda x: x[1]):
                part, start, end = job
                if end > self.max_times[machine]:
                    return False
                if start < current_time:
                    return False
                if prev_color and self.parts_colors[part] != prev_color:
                    if (start - current_time) < self.cleaning_time:
                        return False
                current_time = end
                prev_color = self.parts_colors[part]
        return True

    def evaluate_schedule(self, schedule):

        cleaned = self.add_cleaning_times(schedule)
        total_cost = 0
        for machine in self.machines:
            parts = [job[0] for job in sorted(cleaned[machine], key=lambda x: x[1])]
            for i in range(1, len(parts)):
                p = parts[i-1] - 1
                q = parts[i] - 1
                total_cost += self.alpha[p][q]
        return total_cost

    def visualize(self, interval = 4000):
        cleaned = self.add_cleaning_times(self.schedule) if self.type == "Tabu Search" else self.schedule
        fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        machine_labels = ['Machine 1 Schedule', 'Machine 2 Schedule']
        max_time = max(self.max_times.values())

        for idx, machine in enumerate(self.machines):
            ax = axs[idx]
            for t in range(0, int(max_time) + interval, interval):
                ax.axvline(t, color='grey', linestyle='--', linewidth=0.8)

            for job in sorted(cleaned[machine], key=lambda x: x[1]):
                part, start, end = job
                color = self.parts_colors[part]
                ax.barh(part, end - start, left=start, height=1, color=color, edgecolor='black')
                ax.text((start + end) / 2, part, f'Type {part}', ha='center', va='center', color='white')

            ax.set_title(machine_labels[idx], fontsize=14)
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_yticks(range(1, len(self.parts_colors)+1))
            ax.set_yticklabels([f'Type {i}' for i in range(1, len(self.parts_colors)+1)])
            ax.axvline(self.max_times[machine], color='red', linestyle='--', linewidth=1.5, label=f'Max Time {chr(65 + idx)}')
            ax.legend(loc='upper right')

        fig.suptitle(f'{self.type} Production Schedule for Two Machines', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @abstractmethod
    def solve(self):
        pass

    def __call__(self):
        start_time = time.time()
        solution = self.solve()
        elapsed_time = time.time() - start_time
        return solution, self.objective_value, elapsed_time


class TabuSearchScheduler(BaseScheduler):
    def __init__(self, tabu_tenure=10, max_iterations=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.tabu_list = []
        self.type = "Tabu Search"

    def generate_initial(self):
        schedule = {1: [], 2: []}
        machine_times = {1: 0, 2: 0}
        color_groups = {}
        for part in range(1, self.num_parts + 1):
            color = self.parts_colors[part]
            color_groups.setdefault(color, []).append(part)
        
        for color in color_groups:
            for part in color_groups[color]:
                proc_time = self.calculate_processing_time(part)
                m = min([1, 2], key=lambda x: machine_times[x])
                
                if schedule[m] and self.parts_colors[schedule[m][-1][0]] != color:
                    machine_times[m] += self.cleaning_time
                
                if machine_times[m] + proc_time > self.max_times[m]:
                    m2 = 2 if m == 1 else 1
                    if machine_times[m2] + proc_time > self.max_times[m2]:
                        continue
                    m = m2
                start = machine_times[m]
                end = start + proc_time
                schedule[m].append((part, start, end))
                machine_times[m] = end
        
        return schedule

    def get_neighbors(self, current_schedule):
        neighbors = []
        for machine in self.machines:
            for idx, job in enumerate(current_schedule[machine]):
                part = job[0]
                tgt = 2 if machine == 1 else 1
                neigh = self._create_swap_schedule(current_schedule, part, tgt)
                if neigh and self.validate_schedule(neigh):
                    neighbors.append(neigh)
        for machine in self.machines:
            jobs = current_schedule[machine]
            for i in range(len(jobs)-1):
                neighbor = {
                    1: list(current_schedule[1]),
                    2: list(current_schedule[2])
                }
                neighbor[machine] = list(neighbor[machine])
                neighbor[machine][i], neighbor[machine][i+1] = neighbor[machine][i+1], neighbor[machine][i]
                if self.validate_schedule(neighbor):
                    neighbors.append(neighbor)
        return neighbors

    def _create_swap_schedule(self, schedule, part, target_machine):
        new_schedule = {
            1: [j for j in schedule[1] if j[0] != part],
            2: [j for j in schedule[2] if j[0] != part]
        }
        pt = self.calculate_processing_time(part)
        c = self.parts_colors[part]
        machine_jobs = sorted(new_schedule[target_machine], key=lambda x: x[1])
        best_position = None
        best_start = None
        if machine_jobs:
            last_end = machine_jobs[-1][2]
            if self.parts_colors[machine_jobs[-1][0]] != c:
                last_end += self.cleaning_time
        else:
            last_end = 0
        if last_end + pt <= self.max_times[target_machine]:
            best_position = len(machine_jobs)
            best_start = last_end
        for i in range(len(machine_jobs)):
            prev_end = machine_jobs[i-1][2] if i > 0 else 0
            next_start = machine_jobs[i][1]
            available = next_start - prev_end
            if i > 0 and self.parts_colors[machine_jobs[i-1][0]] != c:
                available -= self.cleaning_time
            if available >= pt:
                st = prev_end
                if i > 0 and self.parts_colors[machine_jobs[i-1][0]] != c:
                    st += self.cleaning_time
                if st + pt <= self.max_times[target_machine]:
                    best_position = i
                    best_start = st
                    break
        if best_position is not None:
            new_job = (part, best_start, best_start + pt)
            new_schedule[target_machine].insert(best_position, new_job)
            return new_schedule
        return None

    def solve(self):
        current_schedule = self.generate_initial()
        best_schedule = current_schedule
        best_cost = self.evaluate_schedule(current_schedule)
        
        for _ in range(self.max_iterations):
            neighbors = self.get_neighbors(current_schedule)
            if not neighbors:
                break
            best_neighbor = None
            best_neighbor_cost = float('inf')
            for neighbor in neighbors:
                h = self._get_move_hash(current_schedule, neighbor)
                if h not in self.tabu_list:
                    cst = self.evaluate_schedule(neighbor)
                    if cst < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cst
            if best_neighbor:
                self.tabu_list.append(self._get_move_hash(current_schedule, best_neighbor))
                if len(self.tabu_list) > self.tabu_tenure:
                    self.tabu_list.pop(0)
                current_schedule = best_neighbor
                if best_neighbor_cost < best_cost:
                    best_schedule = best_neighbor
                    best_cost = best_neighbor_cost
        
        self.schedule = self.add_cleaning_times(best_schedule)
        self.objective_value = best_cost
        return self.schedule

    def _get_move_hash(self, old_schedule, new_schedule):
        moved_parts = []
        for m in self.machines:
            old_parts = {job[0] for job in old_schedule[m]}
            new_parts = {job[0] for job in new_schedule[m]}
            diff = old_parts.symmetric_difference(new_parts)
            moved_parts.extend(diff)
        return tuple(sorted(moved_parts))
