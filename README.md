# Painting Process Optimization

This project uses Gurobi to optimize the painting process for a set of parts, focusing on minimizing color change costs while ensuring efficient scheduling.

## Model Overview

### Parameters
- \( \text{num\_parts} \): Total number of parts to be painted.
- \( \text{demand}_p \): Demand for each part \( p \), where \( p = 1, 2, \dots, \text{num\_parts} \).
- \( \text{parts\_colors}[p] \): Color of each part \( p \), chosen randomly from a set of colors.
- \( \text{colors} \): Set of possible colors \(\{\text{Red, Blue, Green, Yellow, Black, Purple, Orange}\}\).
- \( \text{color\_cost}[c] \): Cost associated with each color \( c \).
- \( \text{unit\_production\_time}_p \): Production time required for each part \( p \), set as 1 for all parts in this example.

### Decision Variables
- \( \text{start\_times}[p] \): Continuous variable representing the start time of part \( p \).
- \( \text{end\_times}[p] \): Continuous variable representing the end time of part \( p \).
- \( \text{order}[p, p'] \): Binary variable, where \( \text{order}[p, p'] = 1 \) if part \( p \) is processed before part \( p' \), and 0 otherwise.
- \( \text{successor}[p, q] \): Binary variable, where \( \text{successor}[p, q] = 1 \) if part \( p \) is immediately followed by part \( q \), and 0 otherwise.
- \( \text{color\_change}[p, p'] \): Binary variable indicating a color change between parts \( p \) and \( p' \); it is 1 if there is a color change and 0 otherwise.

### Objective Function
Minimize the total color change cost:
\[
\min \sum_{p=1}^{\text{num\_parts}} \sum_{q=1}^{\text{num\_parts}} \text{color\_cost}[\text{parts\_colors}[p]] \cdot \text{color\_change}[p, q]
\]

### Constraints
1. **Successor Constraints**:
   - Each part can have at most one immediate successor:
     \[
     \sum_{q \neq p} \text{successor}[p, q] \leq 1 \quad \forall p
     \]
   - Each part can have at most one immediate predecessor:
     \[
     \sum_{p \neq q} \text{successor}[p, q] \leq 1 \quad \forall q
     \]
   - There must be exactly one part without a successor (end part) and one part without a predecessor (start part):
     \[
     \sum_{p} \sum_{q \neq p} \text{successor}[p, q] = \text{num\_parts} - 1
     \]

2. **Mutual Exclusivity**:
   - For each pair of parts \( (p, p') \), one must precede the other:
     \[
     \text{order}[p, p'] + \text{order}[p', p] = 1 \quad \forall p \neq p'
     \]

3. **Order-Successor Link**:
   - Links the `order` and `successor` variables such that if part \( p \) is the successor of part \( p' \), then \( p' \) must come before \( p \) in the sequence:
     \[
     \text{order}[p', p] \geq \text{successor}[p, p'] \quad \forall p \neq p'
     \]

4. **Color Change**:
   - Adds a cost whenever there is a color change between successive parts:
     \[
     \text{color\_change}[p, p'] \geq \text{successor}[p, p'] \quad \text{if} \; \text{parts\_colors}[p] \neq \text{parts\_colors}[p'] \quad \forall p \neq p'
     \]

5. **Demand Fulfillment**:
   - Ensures that each part meets its demand:
     \[
     (\text{end\_times}[p] - \text{start\_times}[p]) \cdot \text{unit\_production\_time}[p] \geq \text{demand}_p \quad \forall p
     \]

6. **No Overlap**:
   - If part \( p \) is scheduled before part \( p' \), then the end time of \( p \) should not overlap with the start time of \( p' \):
     \[
     \text{end\_times}[p] \leq \text{start\_times}[p'] + (1 - \text{order}[p, p']) \cdot M \quad \forall p \neq p'
     \]
     where \( M \) is a large constant.

### Implementation Details
- The model is implemented using Gurobi, with parameters such as `IntFeasTol` and `Threads` set to enhance performance.
- An IIS (Irreducible Inconsistent Subsystem) analysis is available to diagnose infeasibility if the model cannot find a feasible solution.
- Upon finding an optimal solution, the minimized total cost is displayed, and a Gantt chart is generated to visualize the production schedule, color-coded by each part’s color.

### Objective Value
The minimized total cost is displayed once the optimal solution is reached.
