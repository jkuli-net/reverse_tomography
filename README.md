# Reverse Tomography

Tomography can be used to recover a 4d (3d + time) signal from multiple 1d (time) signal recordings.
Reverse Tomography can be used to deliver specific signal recordings to 4d points using 1d signals.

In reverse_tomography.py I demonstrate that 8 audio speakers can deliver completely different customized signals to 8 specific locations.

This has applications for surround sound and directed energy.

fft_reverse_tomography.py is a faster version of same algorithm.
It uses len_signal individual pseudo-inversions sized \[n_rx, n_tx] complex

ct_scan.y simulates an x-ray computed tomography scan. It also simulates a dc-ray reverse tomography. It contains discussion about extending into time dimension for higher frequencies.

fft_reverse_tomography.py MIGHT work with magnets if I increase the signal speed and use the proper cancellation function.
Possible also to adjust the falloff at each point to match the emitter pattern.

It may be possible to accomplish more efficient wireless delivery of power.
That may or may not require highly directional magnets.

Don't call Nikola Tesla yet, this hasn't been tested.

todo: levitation.


Various wave types would behave in different ways with regard to addition and cancellation.
Magnetic fields may leave a directional remainder when they interfere at non parallel field directions.
Light might work for the computation basis but not the quantum basis.
The physics is beyond me at this point.  
I could insert the proper physics for a given wave type and get a more realistic result.

So many types of waves.

How about 3d water?
