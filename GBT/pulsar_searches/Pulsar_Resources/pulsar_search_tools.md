# Pulsar Data Analysis Walkthrough

Each Breakthrough Listen observing session includes one observation of a known pulsar that is used to validate that the system is working properly. A standard pipeline automatically processes these pulsar observations daily. Here we use those pulsar observations as the basis of a walkthrough to introduce new users to the kinds of data analysis used in Breakthrough Listen.

The basic analysis tools are tricky to build, so [Casey Law](http://caseyjlaw.github.io) built an analysis environment using [Docker](http://docker.com). Docker is a bit like a virtual machine and provides a lightweight, portable environment for running software. You might find Casey's introduction to how he's using Docker informative: http://caseyjlaw.github.io/docker-in-astronomy.html.

Steps to get the software running:

1. Ensure you have a working Python installation. Python comes preinstalled on most operating systems. If not, you can find it at https://www.python.org/downloads or via the anaconda software installer at https://www.continuum.io/downloads.
2. Install Docker from https://www.docker.com/products/docker-toolbox
3. In a terminal, install Casey Law's Python tools to access tools on the Docker image:
pip install -e git+https://github.com/caseyjlaw/sidomo.git#egg=sidomo
4. Install the pulsar tools Docker image:
    docker pull caseyjlaw/pulsar-stack
5. Grab one of the GBT Breakthrough Listen pulsar data files by searching the archive at http://breakthroughinitiatives.org/OpenDataSearch. We recommend starting with pulsar data in "filterbank" format. Pulsar observations have "PSR" in their name and filterbank data end in "fil". For example, a file named "blc4_guppi_57407_61054_PSR_J1840+5640_0004.fil":
  - blc4 - Breakthrough Listen Compute node 4 (there are 8 compute nodes numbered from 0 - 7)
  - guppi - the GBT pulsar instrument
  - 57407 - Modified Julian Date of the observation
  - 61054 - number of seconds since start of the day
  - PSR_J1840+5640 - name of the pulsar ("J" refers to Julian epoch, "1840" is the Right Ascension and "5640" is the Declination)
  - 0004 - this is the fourth observation of this source of the day
  - .fil - this is a filterbank file. For an explanation of the different kinds of files, read https://github.com/UCBerkeleySETI/breakthrough/blob/master/GBT/waterfall.md
6. Now run prepfold on the pulsar data. Pulsars emit regular pulses, but these are often too faint to be detected individually, so astronomers need to "fold" the data on the pulsar period. Then various statistics of the pulsar can be determined. This is also a good test of the integrity of the telescope, data, and our systems. 

Change to the directory containing the file you just downloaded and run:

    dodo -- prepfold blc4_guppi_57407_61054_PSR_J1840+5640_0004.fil -psr J1840+5640 -nosearch

It should then spit out a bunch of information about the pulsar, and write several files including a Postscript plot. You can view this directly or convert to PDF. On a Mac you could run something like:

    ps2pdf blc4_guppi_57407_61054_PSR_J1840+5640_0004_PSR_J1840+5640.pfd.ps
    open blc4_guppi_57407_61054_PSR_J1840+5640_0004_PSR_J1840+5640.pfd.pdf

You can see this example file right here in the repository.
