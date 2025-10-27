from tinySA import tinySA
import csv 
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, math
import datetime
import os

xsize=100

#Constant for RBW
RBW = 300000

location = input("Input location number (1-9):")
file_increment = int(input("Input Start Value"))

#file for storing the recorded_channel_measurements
total_power_path = fr'C:\Users\wonde\Documents\Extra_School_files\NSERC Research\TinySA_measurement_stuff\recording_values\location_{location}\total_power.csv'
total_power_file = open(total_power_path, 'a', newline='')

tinySA_obj = tinySA()
tinySA_obj.set_span(30e6)
tinySA_obj = tinySA()
tinySA_obj.set_center(745e6)
tinySA_obj = tinySA()
 
def dbm_to_mwatt(dbm_value):
    return np.pow(10, dbm_value/10)

def mwatt_to_dbm(mwatt_value):
    return 10* np.log10(mwatt_value)

def get_integral(scanned_values):
    delta_x = scanned_values[1][0] - scanned_values[0][0]
    integral = 0

    for value in scanned_values:
        integral += 2 * dbm_to_mwatt(value[1])

    #Subtract one copy of initial and final value
    integral -= dbm_to_mwatt(scanned_values[0][1]) + dbm_to_mwatt(scanned_values[-1][1]) 
    integral *= delta_x/(2*RBW)

    return mwatt_to_dbm(integral)



def get_value():
    global file_increment

    #Get Time
    time = datetime.datetime.now()
    time_string = f"{time.hour:2}:{time.minute}:{time.second}"

    tinySA_obj = tinySA()
    scanned_values = tinySA_obj.scan()
    individual_measurement_path = fr'C:\Users\wonde\Documents\Extra_School_files\NSERC Research\TinySA_measurement_stuff\recording_values\location_{location}\recorded_values_{file_increment}.csv'
    individual_measurement_file = open(individual_measurement_path, 'w', newline='')
    csv.writer(individual_measurement_file).writerow([time_string])
    csv.writer(individual_measurement_file).writerows(scanned_values)
    file_increment+=1

    #Get the Channel Power and write to file
    total_power = get_integral(scanned_values)
    csv.writer(total_power_file).writerow([time_string,total_power])
    total_power_file.flush()
    
    return total_power


while True:
    value = get_value()
    print(f"value = {value}")
    time.sleep(0.7)


"""
def data_gen():
    t = data_gen.t
    while True:
       t+=1
       
       yield t, get_value()

def run(data):
    # update the data
    t,y = data
    if t>-1:
        xdata.append(t)
        ydata.append(y)
        if t>xsize: # Scroll to the left.
            ax.set_xlim(t-xsize, t)
        line.set_data(xdata, ydata)

    return line,

def on_close_figure(event):
    sys.exit(0)

data_gen.t = -1
fig = plt.figure()
fig.canvas.mpl_connect('close_event', on_close_figure)
ax = fig.add_subplot(111)
line, = ax.plot([], [], lw=2)
ax.set_ylim(-100, 0)
ax.set_xlim(0, xsize)
ax.grid()
xdata, ydata = [], []

# Important: Although blit=True makes graphing faster, we need blit=False to prevent
# spurious lines to appear when resizing the stripchart.
ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=500, repeat=False)
plt.show()"""