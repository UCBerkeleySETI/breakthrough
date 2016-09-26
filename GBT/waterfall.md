The current system at GBT is a high speed data recorder, recording voltages as a function of time - the lowest level product you can get from a radio telescope. We also build higher level products, in particular, spectral (detected) data, also known as filterbank format. The link between them is a fast Fourier transform, and then a power computation.

The bandwith (frequency range) we can sample depends on how fast we can sample the voltages. For example, 100 million samples per second implies 50 MHz bandwidth (via the Nyquist criterion). The analog signal from the telescope goes into a digitizer (iADC / iBOB), and then into the SETI processor, which divides the signal up into individual frequency channels, computes the power, and performs a thresholding operation.

The VEGAS instrument at GBT is a big digitizer. It samples at 20 gigasamples / s which gives 10 GHz usable bandwidth. Right now we only run one of VEGAS’s eight ROACH boards, so we get 1/8 of the total bandwidth (about 1.25 GHz). These data come over 10 gigabit ethernet through a network switch to the BL compute infrastructure. Breakthrough Listen will eventually duplicate the existing compute infrastructure by a factor 8, allowing the whole 10 GHz bandwidth to be recorded.

Coarse channelization (using a polyphase filter bank, essentially a big bank of bandpass filters) breaks the incoming band into 256 or 512 pieces.

The Breakthrough machines consist of the head node, storage notes, and compute nodes:

Head node: Contains boot images for the other systems in the cluster.
Storage node (currently 1, will be 8): Long term archival storage with RAID6.
Compute / high speed storage mode (currently 8, will be 64): Where the action happens when we are doing observations. They record raw data to disk. All of the analysis will happen here, in place.

GUPPI (Green bank Ultimate Pulsar Machine) is the old pulsar machine at GBT, that was used for the first SETI observations there. It’s only 800 MHz bandwidth, but it’s the only instrument there currently that can do pulsar timing and has a well-tested baseband capability (i.e. the ability to write raw voltages). The GUPPI software (somewhat modified by Breakthrough engineer Dave MacMahon) is what’s used to record BL data on our new machines.

To generate the high level data products, we take as input a coarsely channelized voltage as a function of time. Output is power as a function of time and frequency, aslo referred to as a waterfall plot. For now let's just look at total intensity (Stokes I), rather than considering the polarization data.

The raw voltages are stored in GUPPI-raw format (also called PSRFITS-raw or “baseband data”).

Information about the file format is at the SETI Brainstorm page at https://seti.berkeley.edu/var/www/html/GBT_SETI_Data_Description . For BL data, the channel ordering is flipped in frequency, and the files are written natively as 8 bit rather than 2 bit (although we're requantizing much of the data to 2 bit after it's taken).

The output from the BL switch is 8 chunks of 64 channels of ~3 MHz width (⅛ of the Nyquist band). Each compute node gets a consecutive chunk in frequency.

Files are stored as one sequence of files per observation per node. There are 64 voltage streams per file. Each file in a sequence is about 18 GB, corresponding to about 20 seconds in time.

Casey Law and collaborators are developing tools to generate waterfall plots from the raw voltage files, which are available in Docker. The output format is “filterbank” (filenames ending in .fil). These have a header that is about 250 bytes, and then a bunch of spectral data, in a sequence of total power spectra from zero up to N.

There are currently four principal code bases. Two pulsar code bases, the GBT spectral line and continuum data reduction code (mostly in IDL), and GBT SETI (there's a github repository for this), which contains the rudiments of the pipeline that we run at GBT. 

You can plot the .fil file using your favorite plotting program (e.g. chop off the header and read the rest in as a binary blob), or you can use some of the sigproc tools to interact with it. For example, if you want to see power as a function of frequency, you can do
bandpass test.fil
