import pandas as pd
import math
import matplotlib.pyplot as plt

C = 899  # heat capacitance in kg / m^3
h = 0.406 # heat transfer coefficient in W / (m^2 * K)
A = [203.0, 638.5, 2970.8, 427.0] # surface area  in m^2
k_vals = []
k_water_vapor = 0.020
house_data = [[], [], [], []]

for i in range(len(A)):
    k = - h * A[i] / C
    k_vals.append(k)
    print(k)


df = pd.read_csv('heatwave_b.csv')
init_temp = df.iloc[0]['Temperature (°F)']

for i in range(len(A)):
    for row in range(len(df)):
        humidity = df.iloc[row]['Humidity (%)'] / 100
        k_norm = (1 - humidity) * k_vals[i] + humidity * k_water_vapor

        cur_temp = df.iloc[row]['Temperature (°F)']
        temp_inside = (init_temp - cur_temp) * math.e ** (k_norm * row) + cur_temp # time in seconds
        print('multiplier: ' + str(math.e ** (k_norm * (row))))
        house_data[i].append(temp_inside)




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