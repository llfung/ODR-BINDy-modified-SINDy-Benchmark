import os
import numpy
# Configure the ranges for i and j as needed
i_values = range(10)  # adjust range for N$i
j_values = range(2)  # adjust range for run_$j

data = numpy.empty(shape=(10,2), dtype='float')

for i in i_values:
    for j in j_values:
        folder_path = os.path.join(f"Time_N{i+1}01", f"run_{j+1}")
        file_path =  os.path.join("./",f"Time_{i+1}_{j+1}.dat")

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                values = abs(float(f.readline()))
            data[i,j] = values
            print(f"Time_N{i+1}/run_{j+1}: {values}")
        else:
            print(f"File not found: {file_path}")
import scipy.io as iomat
results = {'modSINDy_Time':data}
iomat.savemat("modSINDy_Time.mat",results)
