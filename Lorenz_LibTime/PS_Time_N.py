import os
import numpy
# Configure the ranges for i and j as needed
i_values = range(14)  # adjust range for N$i
j_values = range(100)  # adjust range for run_$j

data = numpy.empty(shape=(14,100), dtype='float')

for i in i_values:
    for j in j_values:
        folder_path = os.path.join(f"Lorenz_Lib{i+6}", f"Result")
        file_path =  os.path.join(folder_path,f"Time_Lib{i+6}_{j+1}.dat")

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                values = abs(float(f.readline()))
            data[i,j] = values
            print(f"Time_Lib{i+6}/run_{j+1}: {values}")
        else:
            print(f"File not found: {file_path}")
import scipy.io as iomat
results = {'modSINDy_TimeLib':data}
iomat.savemat("modSINDy_TimeLib.mat",results)
