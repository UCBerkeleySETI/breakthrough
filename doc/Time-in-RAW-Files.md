# Time in Breakthrough Listen RAW files

The best way to get an accurate time from the RAW files is to use a combination
of `STT_IMJD`, `STT_SMJD`, and `PKTIDX`.

  - `STT_IMJD` specifies the date of the start of the scan.  It will be the
    same throughout the scan.

  - `STT_SMJD` specifies the second of the day (0 to 86399, not sure about leap
    seconds).  It also remains the same throughout the scan.

  - `PKTIDX` is the packet index of the first packet of the block.  It should
    increment by 16384 per block, but it may increment by a different value for
    older files.

The time of the start of the block can be calculated using the following
formulae:

    fs = 3e9        # ADC sample rate in Hz
    ts = 1/fs       # ADC sample time in seconds
    tc = 1024 * ts  # Coarse channel sample time in seconds
    tp = 32*tc      # Seconds per packet (32 spectra per packet)

    # This will be a floating point number!
    second_at_start_of_block = STT_SMJD + PKTIDX * tp

    # This will be a floating point number!
    mjd_of_block = STT_IMJD + second_at_start_of_block/86400.0
