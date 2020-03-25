The current system at GBT is a high speed data recorder, recording voltages as a function of time - the lowest level product you can get from a radio telescope. We also build higher level products, in particular, spectral (detected) data, also known as filterbank format. The link between them is a fast Fourier transform, and then a power computation.

The bandwith (frequency range) we can sample depends on how fast we can sample the voltages. For example, 100 million samples per second implies 50 MHz bandwidth (via the Nyquist criterion). The analog signal from the telescope goes into a digitizer (iADC / iBOB), and then into the SETI processor, which divides the signal up into individual frequency channels, computes the power, and performs a thresholding operation.

The VEGAS instrument at GBT is a big digitizer. It samples at 20 gigasamples / s which gives 10 GHz usable bandwidth. Each of VEGAS’s eight ROACH boards handles 1/8 (about 1.25 GHz) of the total bandwidth. These data come over 10 gigabit ethernet through a network switch to the BL compute infrastructure. Breakthrough Listen's 8 banks of 8 compute nodes allow the whole 10 GHz bandwidth from GBT to be recorded.

Coarse channelization (using a polyphase filter bank, essentially a big bank of bandpass filters) breaks the incoming band into 256 or 512 pieces.

The Breakthrough machines consist of the head node, storage notes, and compute nodes:

Head node: Contains boot images for the other systems in the cluster.
Storage nodes (8 machines): Long term archival storage with RAID6.
Compute / high speed storage node (64 machines): Where the action happens when we are doing observations. They record raw data to disk, and after observations are done, perform an FFT to write out the filterbank files (see below).

GUPPI (Green bank Ultimate Pulsar Machine) is the old pulsar machine at GBT, that was used for the first SETI observations there. It’s only 800 MHz bandwidth, but until the BL backend came along it was the only instrument at GBT that could do pulsar timing and had a well-tested baseband capability (i.e. the ability to write raw voltages). The GUPPI software (somewhat modified by Breakthrough engineer Dave MacMahon) is what’s used to record BL data on our new machines.

To generate the high level data products, we take as input a coarsely channelized voltage as a function of time. Output is power as a function of time and frequency, also referred to as a waterfall plot. For now let's just look at total intensity (Stokes I), rather than considering the polarization data.

The raw voltages are stored in GUPPI-raw format (also called PSRFITS-raw or “baseband data”).

The output from the BL switch is 8 chunks of 64 channels of ~3 MHz width (⅛ of the Nyquist band) for each bank. Each compute node gets a consecutive 187.5 MHz chunk in frequency (somewhat confusingly, the highest frequency chunk is the lowest numbered), although the frequency ranges for each compute node overlap somewhat.

Files are stored as one sequence of files per observation per node. There are 64 voltage streams per file. Each file in a sequence is about 18 GB, corresponding to about 20 seconds in time.

Information about the file format, bit depth, and channel ordering are in [Matt Lebofsky's data description paper](https://arxiv.org/pdf/1906.07391.pdf).

Tools are available (https://github.com/UCBerkeleySETI/blimpy) to generate “filterbank” format data from the GUPPI-raw files, and to read the data and headers into Python. Filterbank files store power as a function of frequency and time, and have filenames ending in .fil. These have a header that is about 250 bytes, and then a bunch of spectral data, in a sequence of total power spectra from zero up to N. Blimpy can also produce files in HDF5 format, including files that are compressed using the bitshuffle algorithm.
