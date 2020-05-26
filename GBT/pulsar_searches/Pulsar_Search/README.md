# Pulsar Signal Folding - Breakthrough Listen Data
This notebook illustrates the implementation of pulsar folding and searching techniques from scrath. It finds the period of pusles within fairly noisy data. (Below is a showcase of how "random" pulsar data can appear without proper algorithms)

Checkout the [Notebook Tutorial](https://github.com/PetchMa/Pulsar_Folding/blob/master/Pulsar_folding_Example.ipynb)

<p align="center"> 
    <img src="https://github.com/PetchMa/pulsar_notebooks/blob/master/assets/FAST_folding.gif?raw=true">
</p>

# Fast Fourier and Fast Folding

We first apply the discrete fourier transform onto the data and look for the largest magnitude of the fourier transform. This indicates potential periods within the data. Then we need to check for its consistency and we do so by folding the data by the period the fourier transform indicates.

The folding algorithm is simple. You take each period and you fold the signals ontop of itself. If the period you guessed matches the true period of the pulse then by law of superposition it will increase the signal to noise ratio! This spike in signal to noise ratio can be seen in the following graph.

<p align="center"> 
    <img src="https://github.com/PetchMa/Pulsar_Folding/blob/master/assets/CAN_2.gif?raw=true">
</p>

Depending on the different periods we look at, the pulsar can have multiple profiles. For example with the same observation - instead we look at a period of 0.19 seconds we get a different profile. The collection of profile forms the "finger print" of the pulsar. 

<p align="center"> 
    <img src="https://github.com/PetchMa/Pulsar_Folding/blob/master/assets/can_3.gif?raw=true">
</p>

# Future
Currently looking into building a pulsar timming model. But that will be completed on a later date.