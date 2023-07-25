# Reverse Tomography

Tomography can be used to recover a 4d (3d + time) signal from multiple 1d (time) signal recordings.
Reverse Tomography can be used to deliver specific signal recordings to 4d points using 1d signals.

In fft_reverse_tomography.py I demonstrate that 8 audio speakers can deliver completely different customized signals to 8 specific locations.

This has applications for surround sound and directed energy.

ct_scan.y simulates an x-ray computed tomography scan. It also simulates a dc-ray reverse tomography. It contains discussion about extending into time dimension for higher frequencies.

fft_reverse_tomography.py MIGHT work with magnets if I increase the signal speed and use the proper cancellation function.
Possible also to adjust the falloff at each point to match the emitter pattern.

Various wave types would behave in different ways with regard to addition and cancellation.

I could insert the proper physics for a given wave type and get a more realistic result.

Audio speakers have a 3d falloff function that desribes an out of phase signal behind the speaker, that was not included in the demo.

So many types of waves.

How about 3d water?
