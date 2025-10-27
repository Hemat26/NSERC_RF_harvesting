from tinySA import tinySA
import csv 
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, math
import datetime
import os

#Constant for RBW
RBW = 300000

length = int(input("Input length number (1-3):"))
frequency_number = int(input("Input frequency number (1-3):"))

total_power_path = fr'C:\Users\wonde\Documents\Extra_School_files\NSERC Research\TinySA_measurement_stuff\comparison\location_1\length_{length}\f_{frequency_number}\total_power.csv'
total_power_file = open(total_power_path, 'w', newline='')

def dbm_to_mwatt(dbm_value):
    return np.pow(10, dbm_value/10)

def mwatt_to_dbm(mwatt_value):
    return 10* np.log10(mwatt_value)

def get_integral(scanned_values):
    delta_x = float(scanned_values[1][0]) - float(scanned_values[0][0])
    integral = 0

    for value in scanned_values:
        integral += dbm_to_mwatt(float(value[1]))

    integral *= delta_x/(RBW)

    return mwatt_to_dbm(integral)


num_files = 100
starting_value = int(input("What is the starting value"))


for iteration in range(starting_value, starting_value+num_files):
    individual_measurement_path = fr'C:\Users\wonde\Documents\Extra_School_files\NSERC Research\TinySA_measurement_stuff\comparison\location_1\length_{length}\f_{frequency_number}\recorded_values_{iteration}.csv'
    individual_measurement_file = open(individual_measurement_path, 'r', newline='')
    reader = csv.reader(individual_measurement_file)

    
    time = 0
    values = []
    for i, row in enumerate(reader):
        if(i == 0):
            time = row[0]
        else:
            values.append(row)

    total_energy = get_integral(values)

    csv.writer(total_power_file).writerow([time, total_energy])

