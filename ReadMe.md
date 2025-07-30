# Comparison of modified-SINDy with ODR-BINDy

This repo contain the call scripts to [modified-SINDy](https://github.com/dynamicslab/modified-SINDy) to generate the data for comparison against [ODR-BINDy](https://github.com/llfung/ODR-BINDy).

This is part of a paper on [ODR-BINDy](https://github.com/llfung/ODR-BINDy), which is coming out soon.

### Examples

- Lorenz63
- Rossler
- Van der Pol

## To run

Call the `CX1_modSINDy.pbs` shell scripts to run. It also contain the PBS scheduler instructions to run on a cluster. 

Note that the call script presume some environmental variables (`PBS_O_WORKDIR` and `PBS_ARRAY_INDEX`) to run properly. It also use `module` to set up the Tensorflow environment needed for the package. Adjust them accordingly if you are not running it inside an environment set up by module and PBS.

## Dependencies:

* Numpy, SciPy, Matplotlib, fitter, and TensorFlow 2.0 packages for Python are needed to run the examples.
