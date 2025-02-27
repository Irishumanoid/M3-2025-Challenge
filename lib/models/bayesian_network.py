import math
from pomegranate import *

# bayesian networks and good for modeling complex systems with hidden variables (can use output probabilities to compute expected system output for given inputs)
'''
 (some threshold for each which influences probability of future by normalizing dataset data into categories)
    CO2   Temp  
    |       |      
    |       |      
    _ _ _ _ _ 
         |
       Rain       Wind   
         |         |        
         |         |        
          _ _ _ _ _ 
              |
              |
        Power Outages

'''
co2 = DiscreteDistribution({'low': 0.1, 'medium': 0.5, 'high': 0.4})
temp = DiscreteDistribution({'low': 0.3, 'medium': 0.3, 'high': 0.4})
wind = DiscreteDistribution({'low': 0.1, 'medium': 0.2, 'high': 0.7})

rain = ConditionalProbabilityTable([
    ['low', 'low', 'low', 0.75],  ['low', 'low', 'medium', 0.2],  ['low', 'low', 'high', 0.05],
    ['low', 'medium', 'low', 0.7], ['low', 'medium', 'medium', 0.2], ['low', 'medium', 'high', 0.1],
    ['low', 'high', 'low', 0.65],   ['low', 'high', 'medium', 0.25], ['low', 'high', 'high', 0.1],
    ['medium', 'low', 'low', 0.6],  ['medium', 'low', 'medium', 0.25], ['medium', 'low', 'high', 0.15],
    ['medium', 'medium', 'low', 0.5], ['medium', 'medium', 'medium', 0.3], ['medium', 'medium', 'high', 0.2],
    ['medium', 'high', 'low', 0.45],  ['medium', 'high', 'medium', 0.3], ['medium', 'high', 'high', 0.25],
    ['high', 'low', 'low', 0.45],    ['high', 'low', 'medium', 0.35],  ['high', 'low', 'high', 0.2],
    ['high', 'medium', 'low', 0.4], ['high', 'medium', 'medium', 0.35], ['high', 'medium', 'high', 0.25],
    ['high', 'high', 'low', 0.3],   ['high', 'high', 'medium', 0.4], ['high', 'high', 'high', 0.3]
], [co2, temp]) 


power_outage = ConditionalProbabilityTable([
    ['low', 'low', 'low', 0.95],  ['low', 'low', 'medium', 0.05],  ['low', 'low', 'high', 0.0],
    ['low', 'medium', 'low', 0.85], ['low', 'medium', 'medium', 0.1], ['low', 'medium', 'high', 0.05],
    ['low', 'high', 'low', 0.75],   ['low', 'high', 'medium', 0.2],  ['low', 'high', 'high', 0.05],
    ['medium', 'low', 'low', 0.8],  ['medium', 'low', 'medium', 0.15], ['medium', 'low', 'high', 0.05],
    ['medium', 'medium', 'low', 0.7], ['medium', 'medium', 'medium', 0.2], ['medium', 'medium', 'high', 0.1],
    ['medium', 'high', 'low', 0.6],  ['medium', 'high', 'medium', 0.3], ['medium', 'high', 'high', 0.1],
    ['high', 'low', 'low', 0.55],    ['high', 'low', 'medium', 0.3],  ['high', 'low', 'high', 0.15],
    ['high', 'medium', 'low', 0.45], ['high', 'medium', 'medium', 0.4], ['high', 'medium', 'high', 0.15],
    ['high', 'high', 'low', 0.25],   ['high', 'high', 'medium', 0.35], ['high', 'high', 'high', 0.4]
], [rain, wind]) 


s1 = State(co2, 'co2')
s2 = State(temp, 'temp')
s3 = State(rain, 'rain')
s4 = State(wind, 'wind')
s5 = State(power_outage, 'power outage')

network = BayesianNetwork("Outage prediction")
network.add_states(s1, s2, s3, s4, s5)
network.add_edge(s1, s3)
network.add_edge(s2, s3)
network.add_edge(s3, s5)
network.add_edge(s4, s5)
network.bake()


def get_res(init: map, network: BayesianNetwork):
    beliefs = network.predict_proba(init)
    beliefs = map(str, beliefs)
    print("\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) ))

get_res({'co2': 'low', 'temp': 'high', 'wind': 'low'}, network)
