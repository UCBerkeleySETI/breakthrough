# RAW File Format

The RAW file format is used by the Breakthrough Listen (BL) project to store
channelized voltage from radio telescopes.  It is based on the GUPPI RAW format
used to store pulsar data for the GUPPI project, which is based, loosely, on
the FITS file format.

The basic structure of a RAW file is a series of "header data units" (HDUs).  A
header data unit consists of a header section followed by a data section.  The
header section consists of ASCII text.  The data section is binary.  In some
cases, described below, the header section is followed by padding bytes that
are neither part of the header nor part of the data.  Every header section is
followed by a data section (thereby forming a HDU).  The header section
contains metadata that describe the data section and provide other relevant
details (e.g. time, sky position, frequency, etc.) that correspond to the
voltage samples in the data section.

## RAW header section

A header section consists of a sequence of 80 character ASCII "records".  Each
record is based, loosely, on the FITS header record format.  A record consists
of a keyword (padded with spaces or truncated to 8 characters as necessary)
followed by an equal sign and a space character followed by a value.  The
formatting of the value is also based on FITS conventions.  String values are
enclosed in quotes whereas numeric values are not.  Sometimes numeric values
are stored as strings, so care must be taken to ensure that the values are
parsed correctly.  The final record in the header section is `END` followed by
77 space characters.

### Well known keywords

While applications are free to use any keywords they choose, certain keywords
are predefined to have specific purpose.  The keywords described here are only
a small subset of the keywords that will be found in RAW files, but they
represent the quintessential set of keywords necessary to work with the data in
a RAW file.

  - `BACKEND`

    Certain software packages look for the `BACKEND` keyword to determine the
    file type.  These packages recognize `GUPPI` as meaning GUPPI RAW format.
    To ensure compatibility with these packages, BL RAW files also set
    `BACKEND` to `GUPPI`.

  - `DIRECTIO`

    The BL data recorders use a file writing optimization known as "Direct
    I/O".  This provides streamlined data flow to disk, but it imposes some
    limitations on writes to disk.  The main limitation is that every write to
    disk must be a multiple of 512 bytes.  The number of bytes used by the
    header records is inherently a multiple of 80 bytes, but not not
    necessarily a multiple of 512 bytes.  When using Direct I/O, the header
    records are written with extra padding bytes at the end to ensure that the
    number of bytes written is a multiple of 512.  These extra bytes have no
    significance, but readers must know to skip over them so as not to confuse
    these extra bytes with data.

    Whenever Direct I/O is used, the `DIRECTIO` keyword will be present and set
    to a non-zero value.  This is the indication to the reader that Direct I/O
    was used to write the files and that the header will be followed by padding
    bytes, if necessary.  Readers should check for this keyword.  If it is
    present and non-zero, readers must query file offset after reading the
    `END` record, calculate the number of padding bytes, and seek past them.
    This will position the file pointer to the start of the data section.  If
    the `DIRECTIO` keyword is not present (or present but set to 0), not
    padding bytes are present.

  - `BLOCSIZE`

    The `BLOCSIZE` keyword specifies the total number of bytes in the data
    section that follows this header section.  It does NOT include any of the
    padding bytes that may be present after the header section when `DIRECTIO`
    is non-zero.  The block size can be expressed as:
    
        BLOCSIZE = 2 * NPOL * NTIME * NCHAN * NBITS / 8

  - `NBITS`

    This specifies the number of bits in each complex component per sample.
    Samples in RAW files are complex numbers.  `NBITS` is the number of bits
    used for the real portion and the number of bits used for the imaginary
    portion.  Typical values for `NBITS` are 8, 4, and 2.

    For 8 bit samples, each complex sample consists of two bytes: 1 byte for
    real followed by one byte for imaginary.

    For 4 bit samples, each complex sample consists of one byte with the real
    component in the four most significant bits and the imaginary component in
    the four least significant bits.

    For 2 bit samples, each byte contains two complex samples.  The upper four
    bits contain one sample and the lower four bits contain the other sample.
    For single polarization observations, the upper four bits represent the
    first sample and the lower four bits represent the second sample.  For dual
    polarization observations, the upper four bits represent polarization 0
    (typically X or LCP) and the lower four bits represent polarization 1
    (typically Y or RCP).

  - `NCHAN`

    `NCHAN` represents the number of frequency channels present in the file.

  - `NPOL`

    `NPOL` conatins the number of polarizations present in the subseuent data
    section.  It should be either `1` for single polarization data files or `2`
    for sual polarization data files.  Other values (e.g. 4) may be found in
    some raw files and these errant valuesshould be interpreted as meaning two
    polarizations.

  - `OBSFREQ`

    `OBSFREQ` represents the center frequency of the data contained in the RAW
    file.

  - `OBSBW`
  
    `OBSBW` is the bandwidth covered by the data in the file.  If this is
    negative, then the frequency channels are ordered from highest frequency to
    lowest frequency.

## Data section

A data section follows a header section and any padding bytes necessary when
`DIRECTIO=1`.  The data section contains `BLOCSIZE` bytes that comprise an
array of complex voltage samples.  Raw files always contain complex voltage
samples.  The arrangement of the samples within the data section of a dual
polarization observation is as follows:

    C0T0P0, C0T0P1, C0T1P0, C0T1P1, C1T0P0, ..., CcTtP0, CcTtP1

...where `C0T0P0` represents a complex voltage sample for frequency channel
index 0, time sample 0, polarization 0; `c` is `NCHAN-1`; and `t` is `NTIME-1`.
Note that `NTIME` is usually not present in the header, but can be calculated
as:

    NTIME = BLOCSIZE * 8 / (2 * NPOL * NCHAN * NBITS)

For a single polarization, the data order is essentially the same except that
`P1` samples are obviously not present.

For 8 bit data samples, `C0T0P0` consists of two bytes: one byte for the real
component followed by one byte for the imaginary component.  Each byte is
interpreted as a signed two's complement integer ranging from -128 to +127.

For 4 bit data samples, `C0T0P0` consists of one byte: the upper 4 bits contain
the real component and the lower 4 bits contain the imaginary component.  Each
4 bit nybble is interpreted as a signed two's complement integer ranging from
-8 to +7.

For 2 bit data samples, `C0T0P0` and `C0T0P1` (or, for single polarization
observations, `C0T0P0` and `C0T1P0`) are packed into one byte.  The upper four
bits contain the complex sample for `C0T0P0`; the lower four bits contain the
complex samples for `C0T0P1` (or `C0T1P0).  Within each 4 bit complex value,
the two upper bits are real and the two lower bits are imaginary.  The two bit
values are interpreted as follows:

    00 = +3.3358750
    01 = +1.0
    10 = -1.0
    11 = -3.3358750
