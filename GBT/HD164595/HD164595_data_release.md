September 27, 2016

Dr. Steve Croft

Berkeley SETI Research Center

University of California, Berkeley

The recent claim by a team using the RATAN-600 telescope of a strong signal of possible extraterrestrial origin in the direction of the star HD 164595 (also known as HIP 88194) caused quite a stir in late August. The result was reported in slides scheduled to be presented today, September 27, at the 2016 International Astronautical Congress in Guadalajara, Mexico. Although the transient source appeared bright, it was seen only once. Such one-off detections have a high probability of being radio frequency interference (RFI) from our own human technology, either local to the telescope (cellphones, wifi, radar, etc.), or from a satellite or airplane passing through the telescope’s beam. 

Nevertheless, as a Sunlike star, HD 164595 is a promising target to include in a SETI search, and so the Breakthrough Listen science team at UC Berkeley added the star to our observation schedule. In the early hours of Monday, August 29, we used the 100-meter Green Bank Telescope (GBT) to observe HD 164595 using a nearly identical observing methodology as the original detection. Our analysis showed no radio source associated with the star, to sensitivities almost a factor 100 fainter than the initial claimed transient. We also found no evidence of a transient source at this position in searches of archival data. At the time we wrote up our results in a technical memo (http://seti.berkeley.edu/HD164595.pdf) and a non-technical summary (https://seti.berkeley.edu/HD164595_summary.pdf).

Due to the ubiquity of human-generated RFI, confirmation and localization of claimed signals is an essential component of a SETI search. In addition to the lack of a signal in our GBT observations, a few days later the RATAN team made a public announcement (https://www.sao.ru/Doc-en/SciNews/2016/Sotnikova/) that the transient they had seen was indeed most likely due to RFI.  However, Breakthrough Listen will continue monitoring HD 164595. 

Breakthrough Listen is the most powerful search yet for signatures of extraterrestrial technology, and uses the world’s most powerful telescopes to scan the skies for signs of life. A commitment from Breakthrough Initiatives (https://breakthroughinitiatives.org), the project’s sponsors, is to make as much data as possible available to the public, and the team at Berkeley agrees that openness and transparency are extremely important. Although data volumes are large, and formats are technical, we are today releasing the raw data from our observations of HD 164595 into the Breakthrough Listen archive for independent analysis by anyone with appropriate technical experience.

Most of the GBT data in the archive are in filterbank format (see our github page about data formats (https://github.com/UCBerkeleySETI/breakthrough/blob/master/GBT/waterfall.md), and an iPython notebook demonstrating how to import these files into Python (https://github.com/UCBerkeleySETI/breakthrough/blob/master/GBT/voyager/voyager.ipynb)). Since the HD 164595 observations were undertaken in a raster scanning mode (to match the way that the original RATAN-600 data were taken) rather than our usual "on-off" mode where the telescope alternates between target positions, filterbank files cannot be generated for this particular set of observations. We are therefore releasing the raw “baseband” data for HD 164595 into the archive.

For users who are interested in becoming more familiar with our data products, a good place to start is https://seti.berkeley.edu/listen/ where there are five pages of increasingly detailed information, culminating in a link to our software and to the data archive. Analysis of filterbank files (which are smaller and easier to understand) is a good step to take before downloading the much larger raw files, but for those who wish to access the latter, please read the background material at https://github.com/UCBerkeleySETI/breakthrough/blob/master/doc/RAW-File-Format.md and https://github.com/UCBerkeleySETI/breakthrough/blob/master/doc/Time-in-RAW-Files.md and then search for target name HIP88194 in the Breakthrough Listen archive. Note that the sky positions reported in the headers for this particular target are not correct, due to the raster scan mode used for these observations.

We at Berkeley SETI Research Center, along with our colleagues at other institutions, are excited about the potential of the next few years, as we expand the search for advanced life to a hitherto unprecedented scale. We now know that there are billions of potentially habitable worlds in our own Galaxy alone, and only careful scrutiny can reveal if all are barren and lifeless, or if we in fact have cosmic company.

For more news about Berkeley SETI Research Center, informational videos, and additional tutorials about our data formats and analysis, please follow us on social media:

http://facebook.com/BerkeleySETI

http://twitter.com/setiathome

http://instagram.com/setiathome

http://youtube.com/BerkeleySETI

