import numpy as np

with open("input_INTEL_g2o.g2o", "r") as file:
    
    
    
    for line in file:
        values = line.strip().split(" ")
        type_ver_edge = values[0]
        if type_ver_edge == 'VERTEX_SE2':
            i = values[1]
            x = values[2]
            y = values[3]
            theta = values[4]
            print(type_ver_edge, i, x, y, theta)
        elif type_ver_edge == 'EDGE_SE2':
            print(i)
            i = values[1]
            j = values[2]
            x = values[3]
            y = values[4]
            theta = values[5]
            info = values[6:12]
            print(type_ver_edge, i, j, x, y, theta, info)
            