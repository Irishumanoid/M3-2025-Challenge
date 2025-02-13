from ortools.linear_solver import pywraplp

def begin():
    solver = pywraplp.Solver.CreateSolver('SAT') #Alternatively, use GLOP for LP-only
    if not solver:
        print('Solver creation failed')
        exit()
    return solver

def solve(solver):
    return solver.Solve()

def interpret(result, inputs, objective):
    if result != pywraplp.Solver.OPTIMAL:
        print('The problem does not have an optimal solution.')
        if result == pywraplp.Solver.FEASIBLE:
            print('A potentially suboptimal solution was found.')
        else:
            print('The problem does not have a feasible solution.')
            return
    print('Solution:')
    print("Objective value =", objective.Value())
    for i in range(len(inputs)):
        print('x' + str(i) + '=', inputs[i].solution_value())

def run_lp():
    solver = begin()
    var1 = solver.IntVar(<min>,<max>,name)
    var2 = solver.NumVar(<min>,<max>,name)
    
    constraint1 = v1 >= v2
    solver.Add(constraint1)
    solver.Minimize(<expr>)
    result = solve(solver)
    interpret(result, <list_of_all_input_variables_in_order>, solver.Objective())