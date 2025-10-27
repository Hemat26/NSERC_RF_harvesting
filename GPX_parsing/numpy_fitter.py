import numpy as np


#Assuming that Z = a * x + b * y + c (using positions from the 9 measurement positions)

x = np.array([49.2637,
49.26892,
49.26702,
49.2648,
49.26264,
49.26006,
49.26131,
49.26539,
49.26591])

y = np.array([-123.25619,
-123.25578,
-123.25396,
-123.2532,
-123.25079,
-123.24922,
-123.24669,
-123.24984,
-123.24686])

z1 = np.array([73,
77,
105,
116,
150,
173,
210,
165,
208])

z2 = np.array([244,
363,
320,
270,
223,
165,
192,
283,
294])

A = np.vstack([x, y, np.ones(len(x))]).T


a1, b1, c1 = np.linalg.lstsq(A, z1)[0]
a2, b2, c2 = np.linalg.lstsq(A, z2)[0]

print(a1, b1, c1)
print(a2, b2, c2)


##values from numpy fitting above
def cell_from_lat_long(lat, long):
    return [int(-4.099942947 * lat + 14517.97912 * long + 1789705.009), int(22320.96592 * lat + -5.268575791 * long + -1100017.169)]
