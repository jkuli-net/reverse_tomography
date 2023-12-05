# Reverse Tomography


In fft_reverse_tomography.py I demonstrate that 8 audio speakers can deliver completely different customized signals to 8 specific locations.

This has applications for surround sound and directed energy.

ct_scan.py simulates an x-ray computed tomography scan. It also simulates a dc-ray reverse tomography.

magnet_demo.py simulates a magnet reverse tomography, with 3 dimensional field.

Audio speakers have a 3d falloff function that describes an out of phase signal behind the speaker, that was not included in the demo.

The same function can be used to unmix multichannel recordings if all the positions are known.

