from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the network structure
# Each tuple represents a directed edge from the first element to the second
model = BayesianNetwork([('A', 'C'), ('B', 'C')])
# In this example, A and B are the parents of C, A -> C and B -> C


# Step 2: Define the CPDs

# Variable Card represents the Cardinality, the number of possible states a var
# iable can take. 

# The values are the probabilities of each state. If the variable is affected 
# by a parent, then provide the parents in evidence and evidence_card, and then
# provide the probabilities of each state for each combination of parent states.

# The combination of parent states proceeds in the order:
# The first index represents the output variable cardinality,
# The pseudo-second index (the second and third indices are multiplied, so the 
# large-scale partitioning of the array)
# represents the first parent cardinality, the pseudo-third index represents
# the second parent cardinality (the smaller scale alternating)
# In this example, A and B are the parents of C, so the evidence is ['A', 'B']
# and the evidence_card is [2, 2]


cpd_a = TabularCPD(
    variable='A', 
    variable_card=2, 
    values=[[0.7], [0.3]],
    state_names={'A': ['red', 'blue']}
)
cpd_b = TabularCPD(
    variable='B', 
    variable_card=2, 
    values=[[0.6], [0.4]],
    state_names={'B': ['large', 'small']}
)
cpd_c = TabularCPD(
    variable='C', 
    variable_card=2,
    values=[[0.9, 0.8, 0.7, 0.1],
            [0.1, 0.2, 0.3, 0.9]],
    evidence=['A', 'B'],
    evidence_card=[2, 2], 
    state_names={
    'A': ['red', 'blue'],
    'B': ['large', 'small'],
    'C': ['Ed Sheeran', 'Macklemore']
    }
)

print(cpd_c)

# Step 3: Add the CPDs to the model
model.add_cpds(cpd_a, cpd_b, cpd_c)

# Step 4: Validate the model
assert model.check_model()

# Step 5: Inference
inference = VariableElimination(model)

print("Given A red B large, C?")

posterior1 = inference.query(
    variables=['C'], 
    evidence={'A': 'red', 'B': 'large'}
)

print(posterior1)

print("Given A red, C?")

posterior2 = inference.query(
    variables=['C'], 
    evidence={'A': 'red'}
)

print(posterior2)

print("Given C Ed Sheeran, A?")

posterior3 = inference.query(
    variables=['A'], 
    evidence={'C': 'Ed Sheeran'}
)

print(posterior3)