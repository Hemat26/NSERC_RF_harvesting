#The purpose of this script is to define some helpful functions that are used throughout 
#the rest of the simulations. 


#NOTE: THE DOCUMENTATION IS STILL UNFINISHED (read at your own risk of understanding my code!!!)


#Imports
import sionna.rt
import matplotlib.pyplot as plt
import numpy as np
import math
import drjit as dr
import mitsuba as mi
from typing import Tuple, List
import random 
import matplotlib as mpl
import distinctipy

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies
import warnings
from sionna.rt.utils import watt_to_dbm, log10, rotation_matrix
from sionna.rt.constants import DEFAULT_TRANSMITTER_COLOR, DEFAULT_RECEIVER_COLOR

from tabulate import tabulate



#constants
C = 0.0001 #capacitance
V = 1.8 #voltage
R = 0.1 #series resistance

psens = 0.000001 #(dbm)
psat = 0.001 #(dbm)
a = 42300
b = 5.2

colors_array = distinctipy.get_colors(100)

psens_dbm = watt_to_dbm(psens)
psat_dbm = watt_to_dbm(psat)


#simple dbm_to_watt conversion
def dbm_to_watt(dbm_amount):
    return np.pow(10, dbm_amount/10)/1000


#Function for calculating the radio map from the sum of all the transmitter powers 
#(mostly copied from sionna source code)
def show_sum(
    self,
    metric : str = "rss",
    tx : int | None = None,
    vmin : float | None = None,
    vmax : float | None = None,
    show_tx : bool = True,
    show_rx : bool = False
    ) -> plt.Figure:

    tx_cell_indices = self.tx_cell_indices
    rx_cell_indices = self.rx_cell_indices
    
    tensor = sum_radio_map(self, metric)

    # Convert to dB-scale
    if metric in ["path_gain", "sinr"]:
        with warnings.catch_warnings(record=True) as _:
            # Convert the path gain to dB
            tensor = 10.*log10(tensor)
    else:
        with warnings.catch_warnings(record=True) as _:
            # Convert the signal strengmth to dBm
            tensor = watt_to_dbm(tensor)

    # Set label
    if metric == "path_gain":
        colorbar_label = "Path gain [dB]"
        title = "Path gain"
    elif metric == "rss":
        colorbar_label = "Received signal strength (RSS) [dBm]"
        title = 'RSS'
    else:
        colorbar_label = "Signal-to-interference-plus-noise ratio (SINR)"\
                            " [dB]"
        title = 'SINR'

    # Visualization the radio map
    

    
    fig = plt.figure()
        
    plt.imshow(tensor.numpy(), origin='lower', vmin=vmin, vmax=vmax, animated=True)

    # Set label
    if (tx is None) & (self.num_tx > 1):
        title = 'Sum ' + title + ' across all TXs'
    elif tx is not None:
        title = title + f" for TX '{tx}'"
    plt.colorbar(label=colorbar_label)
    plt.xlabel('Cell index (X-axis)')
    plt.ylabel('Cell index (Y-axis)')
    plt.title(title)

    # Show transmitter, receiver
    if show_tx:
        if tx is not None:
            fig.axes[0].scatter(tx_cell_indices.x[tx],
                                    tx_cell_indices.y[tx],
                                    marker='P',
                                    color=(0,0,0))
        else:
            for tx_ in range(self.num_tx):
                fig.axes[0].scatter(tx_cell_indices.x[tx_],
                                        tx_cell_indices.y[tx_],
                                        marker='P',
                                        color=(0,0,0))

    if show_rx:
        for rx in range(self.num_rx):
            fig.axes[0].scatter(rx_cell_indices.x[rx],
                                    rx_cell_indices.y[rx],
                                    marker='x',
                                    color=(1, 1, 1))
    plt.show()
    return fig


#Function for setting up a matplotlib GIF animation for the radio maps summing each transmitters power
def sum_animation_setup(
    self,
    metric : str = "rss",
    tx : int | None = None,
    vmin : float | None = None,
    vmax : float | None = None,
    show_tx : bool = True,
    show_rx : bool = False,
    fig = None,
    ax = None
    ) -> plt.Figure:

    tx_cell_indices = self.tx_cell_indices
    rx_cell_indices = self.rx_cell_indices
    
    tensor = sum_radio_map(self, metric)

    # Convert to dB-scale
    if metric in ["path_gain", "sinr"]:
        with warnings.catch_warnings(record=True) as _:
            # Convert the path gain to dB
            tensor = 10.*log10(tensor)
    else:
        with warnings.catch_warnings(record=True) as _:
            # Convert the signal strengmth to dBm
            tensor = watt_to_dbm(tensor)

    # Set label
    if metric == "path_gain":
        colorbar_label = "Path gain [dB]"
        title = "Path gain"
    elif metric == "rss":
        colorbar_label = "Received signal strength (RSS) [dBm]"
        title = 'RSS'
    else:
        colorbar_label = "Signal-to-interference-plus-noise ratio (SINR)"\
                            " [dB]"
        title = 'SINR'

    # Visualization the radio map
    
        
    ax.imshow(tensor.numpy(), origin='lower', vmin=vmin, vmax=vmax, animated=True)

    # Set label
    if (tx is None) & (self.num_tx > 1):
        title = 'Sum ' + title + ' across all TXs'
    elif tx is not None:
        title = title + f" for TX '{tx}'"
    #ax.colorbar(label=colorbar_label)
    ax.set_xlabel('Cell index (X-axis)')
    ax.set_ylabel('Cell index (Y-axis)')
    ax.set_title(title)

    # Show transmitter, receiver
    if show_tx:
        if tx is not None:
            ax.scatter(tx_cell_indices.x[tx],
                                    tx_cell_indices.y[tx],
                                    marker='P',
                                    color=(0,0,0))
        else:
            for tx_ in range(self.num_tx):
                ax.scatter(tx_cell_indices.x[tx_],
                                        tx_cell_indices.y[tx_],
                                        marker='P',
                                        color=(0,0,0))

    if show_rx:
        for rx in range(self.num_rx):
            ax.scatter(rx_cell_indices.x[rx],
                                    rx_cell_indices.y[rx],
                                    marker='x',
                                    color=colors_array[rx])
    return fig

#Creates a frame of the 
def sum_animation(
    self,
    metric : str = "rss",
    tx : int | None = None,
    vmin : float | None = None,
    vmax : float | None = None,
    show_tx : bool = True,
    show_rx : bool = False,
    axis = None
    ):
    
    tensor = sum_radio_map(self, metric)

    # Convert to dB-scale
    if metric in ["path_gain", "sinr"]:
        with warnings.catch_warnings(record=True) as _:
            # Convert the path gain to dB
            tensor = 10.*log10(tensor)
    else:
        with warnings.catch_warnings(record=True) as _:
            # Convert the signal strengmth to dBm
            tensor = watt_to_dbm(tensor)
        
    image = axis.imshow(tensor.numpy(), origin='lower', vmin=vmin, vmax=vmax, animated=True)

    rx_cell_indices = self.rx_cell_indices
    for rx in range(self.num_rx):
            axis.scatter(rx_cell_indices.x[rx],
                                    rx_cell_indices.y[rx],
                                    marker='x',
                                    color=colors_array[rx])

    # Set label

    return image


def regions_animation_setup(
    self,
    metric : str = "rss",
    tx : int | None = None,
    vmin : float | None = None,
    vmax : float | None = None,
    show_tx : bool = True,
    show_rx : bool = False,
    fig = None,
    ax = None
    ) -> plt.Figure:

    tx_cell_indices = self.tx_cell_indices
    rx_cell_indices = self.rx_cell_indices
    
    region_map = ["SENS", "RF", "SAT"]

    if metric not in ["path_gain", "rss", "sinr"]:
        raise ValueError("Invalid metric")
    
    sum_map = sum_radio_map(self, metric="rss")
    sum_map = convert_to_region(sum_map)

    # Create the colormap and normalization
    colors = mpl.colormaps['tab20'].colors[:3]
    cmap, norm = from_levels_and_colors(
        list(range(4)), colors)
    ax.imshow(sum_map,
                origin='lower', cmap=cmap, norm=norm)
    ax.set_xlabel('Cell index (X-axis)')
    ax.set_ylabel('Cell index (Y-axis)')
    ax.set_title('Sensitivty-Saturation Map')
    #cbar = ax.colorbar(label="REGION")
    #cbar.ax.get_yaxis().set_ticks([])
    #for region in range(3):
    #    cbar.ax.text(.7, region + .5, region_map[region], ha='center', va='center')

    # Show transmitter, receiver
    if show_tx:
        if tx is not None:
            ax.scatter(tx_cell_indices.x[tx],
                                    tx_cell_indices.y[tx],
                                    marker='P',
                                    color=(0,0,0))
        else:
            for tx_ in range(self.num_tx):
                ax.scatter(tx_cell_indices.x[tx_],
                                        tx_cell_indices.y[tx_],
                                        marker='P',
                                        color=(0,0,0))

    if show_rx:
        for rx in range(self.num_rx):
            ax.scatter(rx_cell_indices.x[rx],
                                    rx_cell_indices.y[rx],
                                    marker='x',
                                    color=colors_array[rx])
    return fig


def regions_animation(
    self,
    metric : str = "rss",
    tx : int | None = None,
    vmin : float | None = None,
    vmax : float | None = None,
    show_tx : bool = True,
    show_rx : bool = False,
    axis = None
    ):
    
    sum_map = sum_radio_map(self, metric="rss")
    sum_map = convert_to_region(sum_map)

    # Create the colormap and normalization
    colors = mpl.colormaps['tab20'].colors[:3]
    cmap, norm = from_levels_and_colors(
        list(range(4)), colors)
    image = axis.imshow(sum_map,
                origin='lower', cmap=cmap, norm=norm)

    rx_cell_indices = self.rx_cell_indices
    for rx in range(self.num_rx):
            axis.scatter(rx_cell_indices.x[rx],
                                    rx_cell_indices.y[rx],
                                    marker='x',
                                    color=colors_array[rx])

    # Set label
    return image

def sum_radio_map(
    self,
    metric : str = "path_gain",
    ) -> mi.TensorXf:
    r"""Returns the radio map values corresponding to transmitter ``tx``
    and a specific ``metric``

    If ``tx`` is `None`, then returns for each cell the maximum value
    accross the transmitters.

    :param metric: Metric for which to return the radio map
    :type metric: "path_gain" | "rss" | "sinr"
    """

    if metric not in ["path_gain", "rss", "sinr"]:
        raise ValueError("Invalid metric")
    tensor = getattr(self, metric)


    # Select metric for a specific transmitter or compute max
    
    tensor = dr.sum(tensor, axis=0)

    return tensor

def sum_cdf(
    self,
    metric : str = "path_gain",
    tx : int | None = None,
    bins : int = 200
    ) -> Tuple[plt.Figure, mi.TensorXf, mi.Float]:
    r"""Computes and visualizes the CDF of a metric of the radio map

    :param metric: Metric to be shown
    :type metric: "path_gain" | "rss" | "sinr"

    :param tx: Index or name of the transmitter for which to show the radio
        map. If `None`, the maximum value over all transmitters for each
        cell is shown.

    :param bins: Number of bins used to compute the CDF

    :return: Figure showing the CDF

    :return: Data points for the chosen metric

    :return: Cummulative probabilities for the data points
    """

    tensor = sum_radio_map(self, metric)
    # Flatten tensor
    tensor = dr.ravel(tensor)

    if metric in ["path_gain", "sinr"]:
        with warnings.catch_warnings(record=True) as _:
            # Convert the path gain to dB
            tensor = 10.*log10(tensor)
    else:
        with warnings.catch_warnings(record=True) as _:
            # Convert the signal strengmth to dBm
            tensor = watt_to_dbm(tensor)

    # Compute the CDF

    # Cells with no coverage are excluded
    active = tensor != float("-inf")
    num_active = dr.count(active)
    # Compute the range
    max_val = dr.max(tensor)
    if max_val == float("inf"):
        raise ValueError("Max value is infinity")
    tensor_ = dr.select(active, tensor, float("inf"))
    min_val = dr.min(tensor_)
    range_val = max_val - min_val
    # Compute the cdf
    ind = mi.UInt(dr.floor((tensor - min_val)*bins/range_val))
    cdf = dr.zeros(mi.UInt, bins)
    dr.scatter_inc(cdf, ind, active)
    cdf = mi.Float(dr.cumsum(cdf))
    cdf /= num_active
    # Values
    x = dr.arange(mi.Float, 1, bins+1)/bins*range_val + min_val

    # Plot the CDF (sike deleted)

    fig = None

    return fig, x, cdf


def concat_point3f(a, b) -> mi.Point3f:
    total = [[],[],[]]

    for i in range(len(a[0])):
        total[0].append(a[0][i])
        total[1].append(a[1][i])
        total[2].append(a[2][i])

    for i in range(len(b[0])):
        total[0].append(b[0][i])
        total[1].append(b[1][i])
        total[2].append(b[2][i])

    return mi.Point3f(total)

import matplotlib as mpl
from matplotlib.colors import from_levels_and_colors

def show_association_more_colours(
        self,
        metric : str = "path_gain",
        show_tx : bool = True,
        show_rx : bool = False
        ) -> plt.Figure:
        r"""Visualizes cell-to-tx association for a given metric

        The positions of the transmitters and receivers are indicated
        by "+" and "x" markers, respectively.

        :param metric: Metric to show
        :type metric: "path_gain" | "rss" | "sinr"

        :param show_tx: If set to `True`, then the position of the transmitters
            are shown.

        :param show_rx: If set to `True`, then the position of the receivers are
            shown.

        :return: Figure showing the cell-to-transmitter association
        """

        tx_cell_indices = self.tx_cell_indices
        rx_cell_indices = self.rx_cell_indices

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        # Create the colormap and normalization
        colors = colors_array[:self.num_tx]
        cmap, norm = from_levels_and_colors(
            list(range(self.num_tx+1)), colors)
        fig_tx = plt.figure()
        image = plt.imshow(self.tx_association(metric).numpy(),
                    origin='lower', cmap=cmap, norm=norm)
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        plt.title('Cell-to-TX association')
        cbar = plt.colorbar(label="TX")
        cbar.ax.get_yaxis().set_ticks([])
        for tx in range(self.num_tx):
            cbar.ax.text(.5, tx + .5, str(tx), ha='center', va='center')

        # Show transmitter, receiver
        if show_tx:
            for tx in range(self.num_tx):
                fig_tx.axes[0].scatter(tx_cell_indices.x[tx],
                                       tx_cell_indices.y[tx],
                                       marker='P',
                                       color=(0,0,0))

        if show_rx:
            for rx in range(self.num_rx):
                fig_tx.axes[0].scatter(rx_cell_indices.x[rx],
                                       rx_cell_indices.y[rx],
                                       marker='x',
                                       color=DEFAULT_RECEIVER_COLOR)
        plt.show()
        return fig_tx, image

#function used for below plotting
def convert_to_region(DC_energy):
    DC_energy = np.where(DC_energy<psens, 0, DC_energy)
    DC_energy = np.where(DC_energy>psat, 2, DC_energy)
    DC_energy = np.where((DC_energy>psens) & (DC_energy<psat),1, DC_energy)
    return DC_energy

#Requires setting the given values for vmin and vmax
def show_DC_regions(
    self,
    metric : str = "path_gain",
    show_tx : bool = True,
    show_rx : bool = False
    ) -> plt.Figure:
        r"""Visualizes cell-to-tx association for a given metric

        The positions of the transmitters and receivers are indicated
        by "+" and "x" markers, respectively.

        :param metric: Metric to show
        :type metric: "path_gain" | "rss" | "sinr"

        :param show_tx: If set to `True`, then the position of the transmitters
            are shown.

        :param show_rx: If set to `True`, then the position of the receivers are
            shown.

        :return: Figure showing the cell-to-transmitter association
        """

        tx_cell_indices = self.tx_cell_indices
        rx_cell_indices = self.rx_cell_indices
        
        region_map = ["SENS", "RF", "SAT"]

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")
        
        sum_map = sum_radio_map(self, metric="rss")
        sum_map = convert_to_region(sum_map)

        # Create the colormap and normalization
        colors = mpl.colormaps['tab20'].colors[:3]
        cmap, norm = from_levels_and_colors(
            list(range(4)), colors)
        fig_tx = plt.figure()
        plt.imshow(sum_map,
                    origin='lower', cmap=cmap, norm=norm)
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        plt.title('Sensitivty-Saturation Map')
        cbar = plt.colorbar(label="REGION")
        cbar.ax.get_yaxis().set_ticks([])
        for region in range(3):
            cbar.ax.text(.5, region + .5, region_map[region], ha='center', va='center')

        # Show transmitter, receiver
        if show_tx:
            for tx in range(self.num_tx):
                fig_tx.axes[0].scatter(tx_cell_indices.x[tx],
                                       tx_cell_indices.y[tx],
                                       marker='P',
                                       color=(0,0,0))

        if show_rx:
            for rx in range(self.num_rx):
                fig_tx.axes[0].scatter(rx_cell_indices.x[rx],
                                       rx_cell_indices.y[rx],
                                       marker='x',
                                       color=DEFAULT_RECEIVER_COLOR)
        plt.show()
        return fig_tx


class RF_harvesting_metrics:
    def __init__(self, _median_energy_RX, _median_energy_DC, _median_charging_time, _first_qart_RX, _third_quart_RX, _mean_energy_RX, _mean_energy_DC, _mean_charging_time, _pdf_power_values, _pdf_values, _sens_percentile, _sat_percentile):
        self.median_energy_RX = _median_energy_RX
        self.median_energy_DC = _median_energy_DC
        self.median_charging_time = _median_charging_time
        self.mean_energy_RX = _mean_energy_RX
        self.mean_energy_DC = _mean_energy_DC
        self.mean_charging_time = _mean_charging_time
        self.pdf_power_values = _pdf_power_values
        self.pdf_values = _pdf_values 
        self.first_qart_RX = _first_qart_RX
        self.third_quart_RX = _third_quart_RX   
        self.sens_percentage = _sens_percentile
        self.sat_percentage = _sat_percentile

    def print_related_values(self):
        first_quart_DC = power_recieved_dbm(self.first_qart_RX)
        first_quart_time = calculate_time(DC_power_recieved(self.first_qart_RX))

        third_quart_DC = power_recieved_dbm(self.third_quart_RX)
        third_quart_time = calculate_time(DC_power_recieved(self.third_quart_RX))

        print(tabulate([['Mean', self.mean_energy_RX, self.mean_energy_DC, self.mean_charging_time], 
                        ['Q1', self.first_qart_RX, first_quart_DC, first_quart_time],
                        ['Median', self.median_energy_RX, self.median_energy_DC, self.median_charging_time],
                        ['Q3', self.third_quart_RX, third_quart_DC, third_quart_time]], 
                        headers=['Type', 'Recieved Power (dbm)', 'DC energy (dbm)', 'time (s)'], 
                        tablefmt='orgtbl'))
        print("\n")
        print(tabulate([[f"{self.sens_percentage*100:.3f}%", f"{(self.sat_percentage-self.sens_percentage)*100:.3f}%", f"{(1-self.sat_percentage)*100:.3f}%"]], headers=['Not enough RF', 'Charging', 'Saturation'], tablefmt='orgtbl'))
        
    def display_cdf_and_pdf(self, cdf_values):
        #CDF is already set to show
        cdf_display_values = []
        for i in range(len(self.pdf_power_values)):
            cdf_display_values.append(cdf_values[i])
        plt.plot(self.pdf_power_values, cdf_display_values);
        plt.grid(True, which="both")
        plt.ylabel("Cummulative probability")
        plt.xlabel("Recieved Power (dbm)")
        plt.title("CDF Chart")
        plt.show()

        plt.plot(self.pdf_power_values, self.pdf_values, color='red')
        plt.grid(True, which="both")
        plt.ylabel("Probability Density")
        plt.xlabel("Recieved Power (dBm)")
        plt.title("Probability Distribution")
        plt.show()


def calculate_slope(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)

#Calculating Power Recieved
def DC_power_recieved(input_power):
    psig = 1 + np.exp(-a * psens + b)
    psig /= 1 + np.exp(-a*input_power + b)
    psig -= 1
    psig *= psat/np.exp(-a * psens + b) 

    result = np.where(input_power<psens, 0, psig)
    result = np.where(input_power > 0.0000251189, input_power * 0.4, result) #efficiency cutoff at 16 dbm
    result = np.where(input_power > 0.001, 0.001 * 0.4, result) #power cutoff at 0 dbm

    return result

def power_recieved_dbm(input_power):
    power_W = math.pow(10, input_power/10)/1000
    power_W = DC_power_recieved(power_W)
    if(power_W == 0):
        return -140
    else:
        return 10 * math.log10(1000 * power_W)
    
#The time taken to charge to a given voltage using a given input power
def calculate_time(input_power,final_voltage=V): #(Basically just equivelent to total stored energy of a capacitor / input power) 
    if(input_power == 0):
        return None

    A = math.sqrt(C*C*final_voltage*final_voltage + 4 * C*C * R * input_power)
    t = math.log((A + C * final_voltage)/(A - C * final_voltage))
    t += (2 * C * final_voltage)/(A - C * final_voltage)
    t *= 0.5 * R * C
    return t

#Calculates the time from a dbm power input
def calculate_time_from_db(power_db):
    power_W = math.pow(10, power_db/10)/1000
    return calculate_time(power_W)


#We can approximate the lambert function as Wo(e^x) = x - ln(x) + ln(x)/x
def lambert_exponential(x):
    return x - np.log(x) + np.log(x)/x

def calculate_voltage_lambert(power, time): 
    exponent_input = 1 + (2 * time)/(R * C)
    Z = 0.5 * (1 + lambert_exponential(exponent_input))
    V = (2 * math.sqrt(R * power) * (1 - 1/Z))/math.sqrt(1 - math.pow(1 - 1/Z, 2))
    return V

#Calculates the final voltage of the capacitor based on a given initial voltage
def calculate_final_voltage(power, time, initial_voltage):
    if(power == 0):
        return initial_voltage
    else:
        initial_time = calculate_time(power, initial_voltage)
        return calculate_voltage_lambert(power, initial_time+time)


def calculate_metrics_from_cdf(cdf_values) -> RF_harvesting_metrics:
    PDF_x_values = []
    PDF_y_values = []

    for i in range(len(cdf_values[1])-1):
        PDF_x_values.append(cdf_values[1][i])

    for i in range(len(cdf_values[1]) - 1):
        slope = calculate_slope(cdf_values[1][i], cdf_values[2][i], cdf_values[1][i+1], cdf_values[2][i+1])
        PDF_y_values.append(slope)


    #Calculating median RX energy and related values
    median_energy_RX = 0
    Q1_energy_RX = 0
    Q3_energy_RX = 0

    found_Q1 = False
    found_median = False
    


    for i in range(len(cdf_values[2])):
        if(not found_Q1):
            if(cdf_values[2][i] >= 0.25):
                found_Q1 = True
                Q1_energy_RX = cdf_values[1][i]

        if(not found_median):
            if (cdf_values[2][i] >= 0.5):
                found_median = True
                median_energy_RX = cdf_values[1][i]
        
        if(cdf_values[2][i] >= 0.75):
            Q3_energy_RX = cdf_values[1][i]
            break

    median_energy_DC = power_recieved_dbm(median_energy_RX)
    median_charging_time = calculate_time(DC_power_recieved(median_energy_RX))

    #Calculating DC regions percentages
    sens_percentile = 1
    sat_percentile = 1
    found_sens = False
    for i in range(len(cdf_values[1])):
        if(cdf_values[1][i] > psens_dbm and (not found_sens)):
            if(i == 0):
                sens_percentile = 0
            else:
                sens_percentile = cdf_values[2][i]
            found_sens = True
        
        if(cdf_values[1][i] > psat_dbm):
            sat_percentile = cdf_values[2][i]
            break

    #Calculating mean RX energy and related values
    area_accumulator = 0
    for i in range(len(PDF_y_values)-1):
        area = 0.5 * (PDF_x_values[i+1] - PDF_x_values[i]) * (PDF_y_values[i+1] * PDF_x_values[i+1] + PDF_y_values[i] * PDF_x_values[i]) 
        area_accumulator += area

    mean_energy_RX = area_accumulator

    mean_energy_DC = power_recieved_dbm(mean_energy_RX)
    mean_charging_time = calculate_time(DC_power_recieved(mean_energy_RX))

    #Returns class containing the values
    return RF_harvesting_metrics(median_energy_RX, median_energy_DC, median_charging_time, Q1_energy_RX, Q3_energy_RX, mean_energy_RX, mean_energy_DC, mean_charging_time, PDF_x_values, PDF_y_values, sens_percentile, sat_percentile)

#compute precoding_vector for rotation beamforming
def compute_precoding_vector(num_cols, steering_angle):
    enumerated = np.arange(num_cols)
    enumerated = np.exp(1j*np.pi*enumerated*np.sin(steering_angle))
    enumerated /= np.sqrt(num_cols)

    return [dr.auto.ad.TensorXf(enumerated.real), dr.auto.ad.TensorXf(enumerated.imag)]
