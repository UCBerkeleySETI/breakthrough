# Getting started with Breakthrough Listen

This document is aimed primarily at students working with us at Berkeley SETI Research Center as part of the Breakthrough Listen [internship program](https://seti.berkeley.edu/Internship.html), but may be of interest to other audiences.

Most of the projects our interns work on require some basic astronomy knowledge as well as some programming skills, typically in Python. When working through the following material, if you are an expert Python programmer, for example, you can skip the basics, but you still may find that some of the material specifically related to how we use Python for our projects is new to you. So please skim through the parts you are familiar with, and focus more on the parts you are not.

The search for extraterrestrial intelligence is a field that bridges observational astronomy, instrument design and engineering, computing, and programming, and touches on other areas such as the search for planets and the basics of biology. Many projects at Berkeley SETI Research Center involve our work with radio telescopes, particularly the Green Bank and Parkes telescopes. Here's what we would like you to know:

## Computer skills

The [O'Reilly](https://www.oreilly.com/) books are a good place to learn about many aspects of modern computing. Some institutions (including UC Berkeley) and public libraries have a subscription to electronic versions of the O'Reilly catalog, as well as other books of interest. Try accessing http://proquest.safaribooksonline.com/ or borrow the physical books from your local library. You'll find lots of books about Unix, Python, SQL, and many other topics.

Before starting work with us, we recommend that you update your operating system to the latest version. For Mac OS, please also install [Xcode](https://developer.apple.com/xcode/) to ensure that you have up to date compilers. You'll also need the Xcode command line tools, which you can install by typing ``xcode-select --install`` in a terminal once Xcode is installed. We also recommend that you install the [MacPorts](https://www.macports.org/install.php) package manager - note the dependencies on Xcode here. 

### Basic Unix skills

Unix is the basis of the operating systems that we use on pretty much all of our computers at Berkeley SETI. If it's not Unix proper, it's a variant such as Linux or MacOS. You'll need to know how to open a terminal window, as well as basic commands like `cd`, `ls`, `mv`, etc. and how to `ssh` into a remote machine as well as how to run a VNC session (many of us use [Chicken](https://sourceforge.net/projects/chicken/) on our Macs for this). You'll also be writing shell scripts as part of your internship.

Here's [a good introduction to doing all this on a Mac](https://developer.apple.com/library/content/documentation/OpenSource/Conceptual/ShellScripting/Introduction/Introduction.html#//apple_ref/doc/uid/TP40004268-TP40003516-SW1).

If you only have a PC running Windows, try installing Ubuntu Linux as a dual-boot system, following the instructions at https://www.ubuntu.com/download/desktop/install-ubuntu-desktop, or install the Windows Subsystem for Linux, https://docs.microsoft.com/en-us/windows/wsl/install-win10. Then follow the Unix tutorial above, or a similar tutorial specifically for Ubuntu (e.g. Chapter 7 of [Ubuntu: Up and Running](http://proquest.safaribooksonline.com/book/operating-systems-and-server-administration/linux/9781449382827)).

You'll also need to learn how to use the `git` version control system, so take a look at https://backlogtool.com/git-guide/en/intro/intro1_1.html

Many folks here edit code in `vi` or `emacs` but another great visual editor to try is [Atom](https://atom.io/).

### Python

There are several O'Reilly books focused on learning Python (e.g. http://proquest.safaribooksonline.com/book/programming/python/9781449355722), or you can learn online at https://www.learnpython.org/, https://www.python.org/about/gettingstarted/, https://www.codecademy.com/tracks/python, or a variety of other sites. Here's a tutorial focused on Python for astronomers: http://python4astronomers.github.io/index.html

If you already have some programming experience, there's a great intro to Python as part of Kaggle's data science curriculum at https://www.kaggle.com/learn/overview - this also covers machine learning, data visualization, SQL, and a bunch of other really useful stuff.

If you don't have Python installed (or even if you do, for example the system Python on OSX), we recommend installing the [Anaconda](https://www.continuum.io/downloads) distribution, which comes with lots of packages that you'll need pre-installed.

Once you are familiar with Python, skills that you'll find useful for our research include familiarity with these packages:
* [numpy](https://docs.scipy.org/doc/numpy/user/index.html)
* [matplotlib](http://matplotlib.org/)
* [Jupyter](http://jupyter.org/)
* [Astropy](http://docs.astropy.org/en/stable/) - particularly "Core data structures and transformations", and "Connecting up: Files and I/O"
* Pandas - for working with array-based data and other data science applications; great tutorial [here](https://www.kaggle.com/learn/pandas), which is part of a Kaggle's [introduction to data science and machine learning tools](https://www.kaggle.com/learn/overview) also linked above

Danny Price also has a nice intro to some of the Python tools we use in our work at https://gist.github.com/telegraphic/790df2b9dc94dcb690053fe563687282

If you are familiar with all of the above, and want to dig a little deeper into visualization tools, you might be interested in [Bokeh](http://bokeh.pydata.org/en/latest/).

### Other languages

Many of our interns are also familiar with C++, Javascript (including D3.js), PHP, R, IDL, Matlab, and other packages. If you are just starting out, we'd suggest learning Python, but if you already have extensive experience in one of the above environments, we encourage you to develop code in the environment in which you are most comfortable.

Common tasks we find ourselves dealing with include interfacing with SQL databases, visualizing data for the web (D3 is nice for that but there are other good options too), manipulating and analyzing data files on the command line, and setting up tools to simplify how various machines talk to each other across the network. Often this involves writing code to glue aspects of code running on a variety of sub-systems together.

## Astronomy basics

Can you explain the following terms in one or two sentences? Right ascension, declination, altitude and azimuth, B1950 and J2000 coordinates, arcseconds, parsecs, Galactic and equatorial coordinates, exoplanet, transit, resolving power, signal-to-noise ratio, luminosity, flux?

If not, pick up an astronomy textbook and do some background reading, or check out an astronomy course online. For example, lectures 1, 4, 11, and 21 from http://ircamera.as.arizona.edu/astr_250/class_syl.html may be useful.

## BL science

Now would be a good time, if you haven't done so already, to read through the basic strategy of the Breakthrough Listen search. The material at https://seti.berkeley.edu/listen/ starts on a basic level; click through pages 1 - 4, and it gets more in-depth as you go. You don't need to read page 5, which duplicates some of the material presented here.

Now you've done a lot of background reading, you probably want to learn [how to find ET](https://github.com/UCBerkeleySETI/breakthrough/blob/master/GBT/README.md). Please read carefully through the material in that document, which describes the BL search. The project you will be working on is a part of this program, and may involve data analysis, data visualization, RFI identification and mitigation, interacting with databases, or other aspects of the search for ET. If you are able to download and display some filterbank files before starting your internship, you'll be off to a great head start.

## Radio astronomy

A good introduction to radio astronomy is at http://www.cv.nrao.edu/course/astr534/ERA_old.shtml - working through chapters 1 - 3 would be useful. There is also a free ebook at https://link.springer.com/book/10.1007%2F978-3-319-44431-4

Both of these radio astronomy primers go into more depth than you will need to know for your summer project, but it's still good to have a handle on the basics. Terms it would be good to be familiar with include Jansky, interferometer, sidelobe, system temperature, spectrometer, correlator, and beamformer.

## Social media, videos, etc.

* One-hour lecture by Steve Croft giving an overview of our search, aimed at general audiences: https://youtu.be/LJaQi8XYzRU
* One-hour astronomy colloquium on Breakthrough Listen by Berkeley SETI Director Andrew Siemion: https://youtu.be/vQ2sKwwhRgI?t=1m48s

We encourage you to follow us on [Facebook](http://www.facebook.com/BerkeleySETI), [Twitter](http://twitter.com/berkeleyseti), [Instagram](http://instagram.com/berkeleyseti), and [YouTube](http://youtube.com/berkeleyseti).
