# Painting Process Optimization

This project uses Gurobi to optimize the painting process for a set of parts, aiming to minimize color change costs and ensure efficient scheduling.

## Model Overview

### Parameters
- `num_parts`: Total number of parts to be painted.
- `demand`: Demand for each part, set as a dictionary with part IDs as keys and demand values (e.g., 4) as values.
- `parts_colors`: A dictionary mapping each part to a specific color, chosen randomly from a predefined set of colors.
- `colors`: List of possible colors (`['Red', 'Blue', 'Green', 'Yellow', 'Black', 'Purple', 'Orange']`).
- `color_cost`: Dictionary mapping each color to a cost.
- `unit_production_time`: Dictionary representing the time required to produce each part, with each part's production time set to 1.

### Decision Variables
- `start_times[p]`: Continuous variable representing the start time of part `p`.
- `end_times[p]`: Continuous variable representing the end time of part `p`.
- `order[p, p']`: Binary variable indicating if part `p` comes before part `p'` in the production sequence.
- `successor[p, q]`: Binary variable indicating if part `p` is immediately followed by part `q` in the production sequence.
- `color_change[p, p']`: Binary variable indicating if there is a color change between parts `p` and `p'`.

### Objective Function
minimize ∑ color_cost[parts_colors[p]] * color_change[p, q]

### Constraints

1. **Successor Constraints**:
   - Each part can have at most one immediate successor:
     ```
     ∑_{q ≠ p} successor[p, q] ≤ 1,  ∀ p
     ```
   - Each part can have at most one immediate predecessor:
     ```
     ∑_{p ≠ q} successor[p, q] ≤ 1,  ∀ q
     ```
   - There must be exactly one part without a successor:
     ```
     ∑_{p} ∑_{q ≠ p} successor[p, q] = num_parts - 1
     ```

2. **Mutual Exclusivity**:
   - For each pair of parts `(p, p')`, one must precede the other:
     ```
     order[p, p'] + order[p', p] = 1, ∀ p ≠ p'
     ```

3. **Order-Successor Link**:
   - If part `p` is the successor of part `p'`, then `p'` must come before `p` in the sequence:
     ```
     order[p', p] ≥ successor[p, p'], ∀ p ≠ p'
     ```

4. **Color Change**:
   - Add a cost whenever there is a color change between successive parts:
     ```
     color_change[p, p'] ≥ successor[p, p'], if parts_colors[p] ≠ parts_colors[p'], ∀ p ≠ p'
     ```

5. **Demand Fulfillment**:
   - Each part must meet its demand:
     ```
     (end_times[p] - start_times[p]) * unit_production_time[p] ≥ demand[p], ∀ p
     ```

6. **No Overlap**:
   - If part `p` is scheduled before part `p'`, then:
     ```
     end_times[p] ≤ start_times[p'] + (1 - order[p, p']) * M, ∀ p ≠ p'
     ```
   where `M` is a large constant.

### Implementation Details
- The model is built and optimized using Gurobi, with optional parameters such as `IntFeasTol` and `Threads` for improved performance.
- If the model is infeasible, an IIS (Irreducible Inconsistent Subsystem) is computed to diagnose infeasibility.
- A Gantt chart is generated to visualize the production schedule for each part, color-coded by the part's color.

### Objective Value
Upon finding an optimal solution, the minimized total cost is displayed.
