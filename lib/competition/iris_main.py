import pandas as pd
import math
import os
import json
import matplotlib.pyplot as plt

C = 899  # heat capacitance in kg / m^3
h = 0.406 # heat transfer coefficient in W / (m^2 * K)
A = [203.0, 638.5, 2970.8, 427.0] # surface area  in m^2
k_vals = []
k_water_vapor = 0.020
house_data = [[], [], [], []]

#Calc k values for each house type given C,h,A
for i in range(len(A)):
    k = - h * A[i] / C
    k_vals.append(k)
    print(k)

#Import heatwave csv over Memphis
#df = pd.read_csv('heatwave_b.csv')

print(os.getcwd())
df = pd.read_csv('./lib/competition/heatwave_b.csv')
init_temp = df.iloc[0]['Temperature (°F)']


for i in range(len(A)):
    for row in range(len(df)):
        humidity = df.iloc[row]['Humidity (%)'] / 100
        k_new = (1 - humidity) * k_vals[i] + humidity * k_water_vapor

        cur_temp = df.iloc[row]['Temperature (°F)']
        temp_inside = (init_temp - cur_temp) * math.e ** (k_new * row) + cur_temp # time in seconds
        house_data[i].append(temp_inside)


house_type_temps = {
    'house1': house_data[0],
    'house2': house_data[1],
    'house3': house_data[2],
    'house4': house_data[3]
}

with open('house_type_temps_over_24h.json', 'w') as file:
    json.dump(house_type_temps, file, indent=4)

fig, axes = plt.subplots(nrows=2, ncols=2)
x = [i for i in range(len(df))]
axes[0, 0].plot(x, house_data[0])
axes[0, 0].plot(x, df['Temperature (°F)'])
axes[0, 0].legend(['Inside temp', 'Outside temp'])
axes[0, 0].set_title('Home 1')
axes[0, 1].plot(x, house_data[1])
axes[0, 1].plot(x, df['Temperature (°F)'])
axes[0, 1].set_title('Home 2')
axes[1, 0].plot(x, house_data[2])
axes[1, 0].plot(x, df['Temperature (°F)'])
axes[1, 0].set_title('Home 3')
axes[1, 1].plot(x, house_data[3])
axes[1, 1].plot(x, df['Temperature (°F)'])
axes[1, 1].set_title('Home 4')
plt.suptitle("Inside home temp (F) vs time since 12 am", fontsize=14)
plt.tight_layout()
plt.show()


