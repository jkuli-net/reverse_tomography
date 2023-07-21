# Reverse Tomography

Tomography can be used to recover a 4d (3d + time) signal from multiple 1d (time) signal recordings.
Reverse Tomography can be used to deliver specific signal recordings to 4d points using 1d signals.

In reverse_tomography.py I demonstrate that 8 audio speakers can deliver completely different customized signals to 8 specific locations.

This has applications for surround sound and directed energy.

fft_reverse_tomography.py is a faster version of same algorithm.
It uses len_signal individual pseudo-inversions sized \[n_rx, n_tx] complex

ct_scan.y simulates an x-ray computed tomography scan. It also simulates a dc-ray reverse tomography. It contains discussion about extending into time dimension for higher frequencies.
