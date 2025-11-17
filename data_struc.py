"""Initialises data structure
"""
import numpy as np

def data_init(t_max, h):
    """ Creates empty data structure as below:
    [
    [[pos_u], [vel_u], [acc_u]]
    [[pos_s], [vel_s], [acc_s]]
    [[time], [input], [unused]]
    ]
    """

    # Sets up empty 3x3x1 array
    n = int(np.ceil(t_max / h))
    new_data = np.zeros((3, 3, n))

    # Inputs forcing curve and time values
    for n in range(len(new_data[2, 0])):
        time = n * h
        new_data[2, 0, n] = time

        # This code defines the shape of the input 'bump'
        # Sets start and end points of input
        if time < 0.7 and time > 0.2:
            # Does the majority of heavy lifting - first term defines height, 0.2 is starting time
            # And 2 is the frequency modifier
            new_data[2, 1, n] = 0.05*np.sin(((new_data[2, 0, n] - 0.2 )* 2 * np.pi))
        else:
            new_data[2, 1, n] = 0


    return new_data
