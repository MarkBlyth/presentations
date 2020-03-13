import scipy
from scipy import signal
import math
import numpy as np
import matplotlib.pyplot as plt


filename = 'neurondata.txt' # where we're loading neuron data from
smoothness = 100            # 1 / cutoff frequency


def load_data(filename):
    # Load a datafile
    # Return an np array of the data
    indata = []
    with open(filename,'r') as infile:
        for line in infile:
            indata.append([float(x) for x in line.split()])
    return np.array(indata).T

def get_freq_spectrum(data, nyquist_freq=1):
    """
    nyquist_freq = half the sample rate;
        highest frequency component we can discern from the sampled data.
    data = 1d array to FFT on
    """
    # If we were given a list, turn it to an array instead
    if isinstance(data, list):
        data = np.array(data)

    # Run FFT
    full_fft = scipy.fft.fft(data)

    # Get the parts of the FFT corresponding to positive frequencies
    positive_parts = full_fft[: math.floor(len(full_fft) / 2)]
    # Find the amplitude of each frequency
    amplitude = np.abs(positive_parts)
    # Figure out what each frequencies is
    freqs = np.linspace(0, nyquist_freq, amplitude.shape[0])

    return amplitude, freqs


def run_LP_filter(data, cutoff_freq, nyquist_freq=None, order=10, filter=signal.butter):
    """
    data = np array that we want to smooth
    cutoff_freq = frequencies above which we want removing
    nyquist_freq = half the sampling frequency (leave it blank if we don't know)
    order = how aggressive we want the cutoff to be
    We can swap filter to signal.chebychev, for example, to use a different filter type.
    
    eg. 
    smoothed_data = run_LP_filter(data, 0.01) # Don't know nyquist freq.
                                               # Keeps first 1% of frequencies, cuts the rest off

    nyquist_freq is the highest frequency we can discern from the data (given by half
    the sampling frequency). The cutoff_freq must be between zero and nyquist_freq.
    Smaller cutoff_freq means we're removing more of the signal and get a smoother output.
    """
    # This just gives us the option of not specifying the nyquist frequency
    fs = 2*nyquist_freq if nyquist_freq else None 
    # Get the parameters needed to build a filter
    filter_params = filter(order, cutoff_freq, "lp", output="sos", fs=fs)
    # Run the filter on the data, and return the smoothed output
    return signal.sosfilt(filter_params, data)


def main():
    # Load the data from the filename specified at the top of this file
    data = load_data(filename)
    # data[0] = sample times
    # data[1] = sample values

    # Filter the current data
    filtered_x = run_LP_filter(data[1],1/smoothness)
    # Whiten the data (original - filtered)
    whitened_x = data[1]-filtered_x
    # Filter the whitened data
    filtered_whitened_x = run_LP_filter(whitened_x, 1/10)

    # We haven't done anything with the frequency spectrum here,
    # but if we wanted to, this is the line of code we would use.
#    amplitude,freqs = get_freq_spectrum(data[1])

    #  Plot the data
    # COMMENT IN/OUT WHICHEVER LINES YOU WANT PLOTTING!
    fig,axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(data[0], data[1],label='Raw data') # totally raw data
#    axarr[0].plot(data[0], filtered_x, label='Filtered data') # smoothed raw data
#    axarr[1].plot(data[0],whitened_x, label='Raw zero\'d data') # unsmoothed whitened data
    axarr[1].plot(data[0],filtered_whitened_x, label='Smoothed zero\'d data') # smoothed whitened data

    # Pretty up the plots
    axarr[0].set_title("Raw data")
    axarr[1].set_title("Zero'd data")
    axarr[0].set_ylabel('Current')
    axarr[1].set_ylabel('Residual')
    axarr[1].set_xlabel('Time')
    axarr[0].legend()
    axarr[1].legend()
    fig.show()

if __name__ == "__main__":
    # Fancy way of saying run the main function when we run this python file.
    # Allows us to import this file as a module if we want to use its functions elsewhere.
    main()
