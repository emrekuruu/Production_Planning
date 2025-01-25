import random
import time
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import time 

class BaseScheduler(ABC):
    def __init__(self, num_parts, group_length, max_time_machine_A, max_time_machine_B,
                 demand, color_cost, parts_colors, alpha, unit_production_time, 
                 cleaning_time, machines):
        # Problem parameters
        self.num_parts = num_parts
        self.group_length = group_length
        self.max_times = {1: max_time_machine_A, 2: max_time_machine_B}
        self.demand = demand
        self.color_cost = color_cost
        self.parts_colors = parts_colors
        self.alpha = alpha
        self.unit_production_time = unit_production_time
        self.cleaning_time = cleaning_time
        self.machines = machines
        self.type = None
        
        # Solution storage
        self.schedule = {1: [], 2: []}  # {machine: [(part, start, end)]}
        self.objective_value = 0

    def calculate_processing_time(self, part):
        return self.demand[part] * self.unit_production_time

    def validate_schedule(self, schedule):
        for machine in self.machines:
            current_time = 0
            prev_color = None
            for job in sorted(schedule[machine], key=lambda x: x[1]):
                
                part, start, end = job
                
                # Check machine time limit
                if end > self.max_times[machine]:
                    return False
                
                # Check temporal consistency
                if start < current_time:
                    return False
                
                # Check cleaning time
                if prev_color and self.parts_colors[part] != prev_color:
                    if (current_time - start) < self.cleaning_time:
                        return False
                    
                current_time = end
                prev_color = self.parts_colors[part]
        
        return True

    def evaluate_schedule(self, schedule):
        total_cost = 0
        for machine in self.machines:
            parts = [job[0] for job in sorted(schedule[machine], key=lambda x: x[1])]
            for i in range(1, len(parts)):
                p = parts[i-1] - 1  # Convert to 0-based index for alpha
                q = parts[i] - 1
                total_cost += self.alpha[p][q]
        return total_cost

    def visualize(self, interval = 4000):
        fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        machine_labels = ['Machine 1 Schedule', 'Machine 2 Schedule']

        max_time = max(self.max_times.values())  # Get the maximum time to determine the range

        for idx, machine in enumerate(self.machines):
            ax = axs[idx]

            # Add dashed grey lines at 4000 intervals
            for t in range(0, int(max_time) + interval, interval):
                ax.axvline(t, color='grey', linestyle='--', linewidth=0.8)

            # Iterate through jobs and plot them at their respective product type positions
            for job in sorted(self.schedule[machine], key=lambda x: x[1]):
                part, start, end = job
                color = self.parts_colors[part]  # Fetch color for the part
                ax.barh(part, end - start, left=start, height=1, color=color, edgecolor='black')
                ax.text((start + end) / 2, part, f'Type {part}', ha='center', va='center', color='white')

            # Set up titles, labels, and max time line
            ax.set_title(machine_labels[idx], fontsize=14)
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_yticks(range(1, 11))  # Ensure y-axis represents product types 1 to 10
            ax.set_yticklabels([f'Type {i}' for i in range(1, 11)])
            ax.axvline(self.max_times[machine], color='red', linestyle='--', linewidth=1.5, label=f'Max Time {chr(65 + idx)}')
            ax.legend()

        # Add a global title
        fig.suptitle(f'{self.type} Production Schedule for Two Machines', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the global title
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
        
        # Group parts by color
        color_groups = {}
        for part in range(1, self.num_parts + 1):
            color = self.parts_colors[part]
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(part)
        
        # Schedule color groups together
        for color in color_groups:
            for part in color_groups[color]:
                proc_time = self.calculate_processing_time(part)
                machine = min([1, 2], key=lambda m: machine_times[m])
                
                # Check if we need cleaning time
                if schedule[machine] and self.parts_colors[schedule[machine][-1][0]] != color:
                    machine_times[machine] += self.cleaning_time
                
                # Check machine capacity
                if machine_times[machine] + proc_time > self.max_times[machine]:
                    machine = 2 if machine == 1 else 1
                    if machine_times[machine] + proc_time > self.max_times[machine]:
                        continue
                
                start = machine_times[machine]
                end = start + proc_time
                schedule[machine].append((part, start, end))
                machine_times[machine] = end
        
        return schedule

    def get_neighbors(self, current_schedule):
        neighbors = []
        
        # Generate machine swap neighbors
        for machine in self.machines:
            for idx, job in enumerate(current_schedule[machine]):
                part = job[0]
                target_machine = 2 if machine == 1 else 1
                neighbor = self._create_swap_schedule(current_schedule, part, target_machine)
                if neighbor and self.validate_schedule(neighbor):
                    neighbors.append(neighbor)
        
        # Generate sequence swap neighbors
        for machine in self.machines:
            jobs = current_schedule[machine]
            for i in range(len(jobs)-1):
                neighbor = {
                    1: list(current_schedule[1]),
                    2: list(current_schedule[2])
                }
                neighbor[machine][i], neighbor[machine][i+1] = neighbor[machine][i+1], neighbor[machine][i]
                if self.validate_schedule(neighbor):
                    neighbors.append(neighbor)
        
        return neighbors

    def _create_swap_schedule(self, schedule, part, target_machine):
        new_schedule = {
            1: [job for job in schedule[1] if job[0] != part],
            2: [job for job in schedule[2] if job[0] != part]
        }
        
        proc_time = self.calculate_processing_time(part)
        color = self.parts_colors[part]
        machine_jobs = sorted(new_schedule[target_machine], key=lambda x: x[1])
        
        # Find best insertion point
        best_position = None
        best_start = None
        
        # Try inserting at the end
        if machine_jobs:
            last_end = machine_jobs[-1][2]
            if self.parts_colors[machine_jobs[-1][0]] != color:
                last_end += self.cleaning_time
        else:
            last_end = 0
            
        if last_end + proc_time <= self.max_times[target_machine]:
            best_position = len(machine_jobs)
            best_start = last_end
        
        # Check intermediate positions
        for i in range(len(machine_jobs)):
            prev_end = machine_jobs[i-1][2] if i > 0 else 0
            next_start = machine_jobs[i][1]
            
            available = next_start - prev_end
            if i > 0 and self.parts_colors[machine_jobs[i-1][0]] != color:
                available -= self.cleaning_time
                
            if available >= proc_time:
                start = prev_end
                if i > 0 and self.parts_colors[machine_jobs[i-1][0]] != color:
                    start += self.cleaning_time
                if start + proc_time <= self.max_times[target_machine]:
                    best_position = i
                    best_start = start
                    break
        
        if best_position is not None:
            new_job = (part, best_start, best_start + proc_time)
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
                
            # Find best non-tabu neighbor
            best_neighbor = None
            best_neighbor_cost = float('inf')
            
            for neighbor in neighbors:
                move_hash = self._get_move_hash(current_schedule, neighbor)
                if move_hash not in self.tabu_list:
                    cost = self.evaluate_schedule(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
            
            if best_neighbor:
                # Update tabu list
                move_hash = self._get_move_hash(current_schedule, best_neighbor)
                self.tabu_list.append(move_hash)
                if len(self.tabu_list) > self.tabu_tenure:
                    self.tabu_list.pop(0)
                
                current_schedule = best_neighbor
                if best_neighbor_cost < best_cost:
                    best_schedule = best_neighbor
                    best_cost = best_neighbor_cost
        
        self.schedule = best_schedule
        self.objective_value = best_cost
        return best_schedule

    def _get_move_hash(self, old_schedule, new_schedule):
        moved_parts = []
        for machine in self.machines:
            old_parts = {job[0] for job in old_schedule[machine]}
            new_parts = {job[0] for job in new_schedule[machine]}
            moved_parts.extend(list(old_parts.symmetric_difference(new_parts)))
        return tuple(sorted(moved_parts))