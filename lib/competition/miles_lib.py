from random import *
seed = 0

neg_inf = float('-inf')
pos_inf = float('inf')

#Copilot, write a lininterpolation function

def lininterpolation(x, x0, x1, y0, y1):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

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
 
 
def demographic_restorative_coefficient(median_household_income, percent_nonwhite)
    inc_part = 20_000 / min(median_household_income, 20_000)
    race_part = 1 - percent_nonwhite
    return inc_part + race_part

#TODO: Get full population demographic instead of household count?
def age_risk(percent_household_over_65, percent_household_under_18):
    pass

def sample_age_category(pct_over_80, pct_between_65_80, pct_between_65_16, pct_below_16): #Last percent is below 65 so that's obvious
    val = random()
    if val < pct_over_80:
        return "Elderly"
    elif val < pct_over_80 + pct_between_65_80:
        return "Retired"
    elif val < pct_over_80 + pct_between_65_80 + pct_between_65_16:
        return "Working Adult"
    else:
        return "Child"

def age_risk(age_category): #How tf do I normalize this??? don't I do proportion to total deaths? but then that reduces the whole thing?? so I have to do proportion to total population?? wait oh this is like chisq counts
    if age_category == "Elderly":
        return 10
    elif age_category == "Retired":
        return 1
    elif age_category == "Working Adult":
        return 0.1
    else:
        return 0.1

def sample_is_working(age_category):
    # Bayesian Math here
    

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
    if commute_type == "wfh":
        #Hour 0 to 23:
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

def shade_temp_amount():
    pass

def solve_PHOV():
    pass

def sample_family_size(house_type):
    pass

def sample_k():
    pass



   
#def working_at_place_