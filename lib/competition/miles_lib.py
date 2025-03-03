from random import *
import json
import pandas as pd
import os

house_type_temps_over_24h = None

with open('house_type_temps_over_24h.json', 'r') as file:
    house_type_temps_over_24h = json.load(file)
    
corresponding_age_ranges = [[0,5], [5,10], [10,15], [15,20], [20,25], [25,30], [30,35], [35,40], [40,45], [45,50], [50,55], [55,60], [60,65], [65,70], [70,75], [75,80], [80,85], [85,200]]
    
#This data was exported by R
heatwave_data = pd.read_csv('heatwave_b.csv')

#Get outside temperature per hour
temp_outside = heatwave_data['Temperature (°F)']

neg_inf = float('-inf')
pos_inf = float('inf')

def vectorize_min(list1, list2):
    for i in range(len(list1)):
        if list1[i] == None:
            list1[i] = pos_inf
        if list2[i] == None:
            list2[i] = pos_inf
            
    return [min(list1[i], list2[i]) for i in range(len(list1))]

def vectorize_sum(list1, list2):
    for i in range(len(list1)):
        if list1[i] == None:
            list1[i] = 0
        if list2[i] == None:
            list2[i] = 0
            
    return [list1[i] + list2[i] for i in range(len(list1))]

def vectorize(fun, list):
    return [fun(x) for x in list]

def lininterpolation(x, x0, x1, y0, y1):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

def normalize_probabilities(probs):
    total = sum(probs)
    if total == 0:
        raise ValueError("Sum of probabilities is zero; cannot normalize.")
    return [p / total for p in probs]

def shade(house_type):
    if house_type == "house1":
        return 10
    elif house_type == "house2":
        return 5
    elif house_type == "house3":
        return 0
    elif house_type == "house4":
        return 0


def heat_danger_discrete(heat_index):
    global neg_inf, pos_inf
    if neg_inf <= heat_index <= 80:
        return 0
    elif 80 < heat_index < 90:
        return 1
    elif 90 < heat_index < 105:
        return 2
    elif 105 < heat_index < 130:
        return 3
    elif 130 < heat_index < pos_inf:
        return 4
    else:
        return -1
            
def heatIndex(temperature,relative_humidity):
    return (
        -42.379 + 
        2.049 * temperature + 
        10.143*relative_humidity - 
        .225*temperature*relative_humidity - 
        .006837*temperature*temperature - 
        .0548*relative_humidity*relative_humidity + 
        .001229*temperature*temperature*relative_humidity + 
        .000853*temperature*relative_humidity*relative_humidity - 
        .00000199*temperature*temperature*relative_humidity*relative_humidity
    ) #Model from https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
            
def heat_danger_continuous(heat_index):
    global neg_inf, pos_inf
    if neg_inf < heat_index < 75:
        return 0
    elif 75 < heat_index < 80:
        return lininterpolation(heat_index, 75, 80, 0, 1)
    elif 80 < heat_index < 90:
        return lininterpolation(heat_index, 80, 90, 1, 2)
    elif 90 < heat_index < 105:
        return lininterpolation(heat_index, 90, 105, 2, 3)
    elif 105 < heat_index < 130:
        return lininterpolation(heat_index, 105, 130, 3, 4)
    elif 130 < heat_index < pos_inf:
        return 4
    else:
        return -1
 
 
def demographic_restorative_coefficient(median_household_income):
    inc_part = 20_000 / min(median_household_income, 20_000)
    return inc_part 

#TODO: Get full population demographic instead of household count?
def age_risk(percent_household_over_65, percent_household_under_18):
    pass

def sample_age_category(probabilities, ranges): #Last percent is below 65 so that's obvious
    val = random()
    cumprob = 0
    for i in range(len(probabilities)):
        if val < cumprob + probabilities[i]:
            return ranges[i]
        cumprob += probabilities[i]

def sample_house_size(house_type):
    if house_type == "house1":
        return 3
    elif house_type == "house2":
        return 3
    elif house_type == "house3":
        return 2
    elif house_type == "house4":
        return 6

def age_risk(age_category): #How tf do I normalize this??? don't I do proportion to total deaths? but then that reduces the whole thing?? so I have to do proportion to total population?? wait oh this is like chisq counts
    if age_category == "Elderly":
        return 10
    elif age_category == "Retired":
        return 1
    elif age_category == "Working Adult":
        return 0.1
    else:
        return 0.1

def sample_working_status(age_cat_working_probability):
    #to_string_indices = ['Under.5', 'X.5...9', 'X.10...14', 'X15.19', 'X20.24', 'X25.29', 'X30.34', 'X35.39', 'X40.44', 'X45.49', 'X50.54', 'X55.59', 'X60.64', 'X65.69', 'X70.74', 'X75.79', 'X80.84', 'X85.Plus']
    #i = corresponding_age_ranges.index(age_category)
    
    val = random()
    if val < age_cat_working_probability:
        return True
    else:
        return False
    

def sample_commute_type(wfh_pct, public_pct, car_pct = None):
    val = random()
    if val < wfh_pct:
        return "wfh"
    elif val < wfh_pct + public_pct:
        return "public"
    else:
        return "car"

def sample_have_car(commute_type, pct_household_car):
    if commute_type == "car" or pct_household_car > random():
        return True
    else:
        return False
    
def car_hd_mask(has_car):
    if has_car:
        return 2
    else:
        return None

def temp_mask_by_commute_type(commute_type):
    #Hour 0 to 23:
    if commute_type == "no_work":
        return [
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None
            ]
    if commute_type == "wfh":
        return [None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None]
    elif commute_type == "public":
        return [None, None, None, None, None, None, None, None,
                70,   70,   70,   70,   70,   70,   70,   70,
                70,   None, None, None, None, None, None, None]
    elif commute_type == "car":
        return [None, None, None, None, None, None, None, None,
                70,   70,   70,   70,   70,   70,   70,   70,
                70,   70,   None, None, None, None, None, None]

def shade_temp_amount(house_type):
    if house_type == "house1":
        return 10
    elif house_type == "house2":
        return 5
    elif house_type == "house3":
        return 0
    elif house_type == "house4":
        return 0
    
def inside_temp_data(house_type):
    return house_type_temps_over_24h[house_type]

def schedule_temp_data_by_person(commute_type, inside_temp_data):
    return vectorize_min(inside_temp_data, temp_mask_by_commute_type(commute_type=commute_type))

def sample_house_type(house_1_pct, house_2_pct, house_3_pct, house_4_pct):
    # We estimate these percentages from the provided data of number of apartments, houses, townhouses, and other.
    # We only sample from the four typical dwellings listed in the data.
    val = random()
    if val < house_1_pct:
        return "house1"
    elif val < house_1_pct + house_2_pct:
        return "house2"
    elif val < house_1_pct + house_2_pct + house_3_pct:
        return "house3"
    else:
        return "house4"


def sample_family_size(house_type):
    pass

    
def integral_of_PHOV(heat_danger_over_time):
    timescount = len(heat_danger_over_time) #Supposed to be 24
    return sum([heat_danger_over_time[i] for i in range(timescount)]) / timescount


   
#def working_at_place_