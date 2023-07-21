import torch

# delay_matrix_from_positions
# return the delay[iref,itar] in samples
def delay_matrix_from_positions(rx_positions, tx_positions, sample_rate, signal_speed):
    rx_pos = rx_positions[:,None,:]
    tx_pos = tx_positions[None,:,:]
    d = torch.sqrt(torch.sum((tx_pos - rx_pos) ** 2, 2)) * sample_rate / signal_speed
    return d

def falloff_matrix_from_positions(rx_positions, tx_positions):    
    # I assume for now the initial volume was measured at distance 1
    falloff_per_double = 0.5011872336272722 # -6db 
    rx_pos = rx_positions[:,None,:]
    tx_pos = tx_positions[None,:,:]
    distance = torch.sqrt(torch.sum((tx_pos - rx_pos) ** 2, 2)) 
    falloff = falloff_per_double ** torch.log2(distance / 1)
    return falloff 
     
# shift the signal by a fractional amount using linear resample
def shift_signal(signal, t):
    f_t = t - torch.floor(t)
    i_t = int(torch.floor(t))
    return torch.roll(signal, i_t, 0) * (1-f_t) + torch.roll(signal, i_t+1, 0) * f_t
       
# delay_fft
# if given an integer, returns the fft of a vector with 1.0 in that position
def delay_fft(delay, data_len, real=False):
    if real:
        fft_len = data_len // 2 + 1
    else:
        fft_len = data_len
    v_delay = delay * -2.0 * torch.pi * torch.arange(fft_len) / data_len
    return torch.exp(1j * v_delay)

# solve_mixed_transmission
# provide positions of transmitters and recievers
# provide the audio you would like each reciever to experience in rx
# returns the proper audio to transmit
def solve_mixed_transmission(n_tx, n_rx, n_data, sample_rate, signal_speed,
                               tx_pos, rx_pos, rx, calc_error=False, use_linear_resample=False):
    
    rx_fft = torch.fft.rfft(rx)

    delay = delay_matrix_from_positions(rx_pos, tx_pos, sample_rate, signal_speed)
    falloff = falloff_matrix_from_positions(rx_pos, tx_pos)
    
    # fft based fractional delay
    a_fft = delay_fft(delay[:,:,None], n_data, real=True) * falloff[:,:,None]

    if use_linear_resample:
        # this would simulate a linear resampled fractional delay
        # matching shift_signal()
        delay0 = delay.int()
        delay1 = delay0 + 1
        fdelay = (delay - delay0)[:,:,None]
        
        a_fft0 = delay_fft(delay0[:,:,None], n_data, real=True)
        a_fft1 = delay_fft(delay1[:,:,None], n_data, real=True)
        a_fft = (a_fft0 * (1.0 - fdelay) + a_fft1 * fdelay) * falloff[:,:,None]

    # pinv needs shape [fft_len, n_rx, n_tx]
    # to broadcast pinv across fft_len
    inv_a_fft = torch.transpose(a_fft, 0, 2)
    inv_a_fft = torch.transpose(inv_a_fft, 1, 2)
    inv_a_fft = torch.linalg.pinv(inv_a_fft)
    inv_a_fft = torch.transpose(inv_a_fft, 0, 2)
    inv_a_fft = torch.transpose(inv_a_fft, 0, 1)

    # broadcasted matmul [n_tx, n_rx] across fft_len
    tx_fft = inv_a_fft * rx_fft[None,:,:]        
    tx_fft = torch.sum(tx_fft, dim=1)

    tx = torch.fft.irfft(tx_fft, n=rx.shape[-1])
    
    if calc_error:        
        _rx_fft = a_fft * tx_fft[None,:,:]
        _rx_fft = torch.sum(_rx_fft, dim=1)
        _rx = torch.fft.irfft(_rx_fft, n=rx.shape[-1])
        abs_error = torch.abs(rx - _rx)
        print('max abs error', torch.amax(abs_error))
        print('mean abs error', torch.mean(abs_error))
        print('min abs error', torch.amin(abs_error))

    return tx

n_tx = 8
n_rx = 8
n_data = 1024

sample_rate = 44100
signal_speed = 343.3    #speed of sound m/sec

# choose any positions, and any desired audio to be recieved at rx_pos

tx_pos = torch.rand(n_tx, 3)
rx_pos = torch.rand(n_rx, 3)
rx = torch.rand( (n_rx, n_data) )

tx = solve_mixed_transmission(n_tx, n_rx, n_data, sample_rate, signal_speed, tx_pos, rx_pos, rx, calc_error=True)

# transmit tx from tx_pos, to recieve rx at rx_pos
