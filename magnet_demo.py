import torch

# do not try this at home

# This is a math demonstration, these are not actual numbers

# Uses multiple magnets to create specially crafted magnetic field.  
# For wireless charging, etc., cover the unused falloff range with zero?

# https://en.wikipedia.org/wiki/Biot%E2%80%93Savart_law

# Biot-Savart_law
# magnetic field for a length L iron bar electromagnet on Z axis with radius R
# I = 1.5  # Current in amperes
# mu_0 = 1.25663706212e-6    # Vacuum permeability  kg⋅m⋅s−2·A−2
# N = 300  # Number of turns in the coil
# L = 0.2  # Length of the iron bar core in meters
# R = 0.02 # Radius of the iron bar core in meters
# r = np.array([x, y, z])  # Observation point
# steps = 10000  # number of integration steps
# mu_r_iron = 1000  # Relative permeability of iron bar 
def magnetic_field_iron_bar_core(r, I, mu_0, N, L, R, steps = 10000, mu_r_iron = 1000):
        
    mu_iron = mu_0 * mu_r_iron    
    wire_length = torch.pi * 2.0 * R * N     # pi * circumference * number_of_windings
    dR_length = wire_length / steps          # change in length per integration segment    
    dL_length = L / steps                    # change in Z pos of wire and bar per integration segment    
    L_offset = L * -0.5                      # Z pos of wire at integration segment zero
    d_radians = (torch.pi * 2 * N) / steps   # change in radians per integration segment
        
    step = torch.arange(steps)
    radians = step * d_radians
    cosines = torch.cos(radians)
    sines = torch.sin(radians)
    coil_pos = torch.empty( (steps, 3))
    coil_vector = torch.empty( (steps, 3))
    coil_pos[:, 0] = cosines * R
    coil_pos[:, 1] = sines * R
    coil_pos[:, 2] = step * dL_length + L_offset
    coil_vector[:, 0] = -sines
    coil_vector[:, 1] = cosines
    coil_vector[:, 2] = 0.0
    r_prime = r[None,:] - coil_pos
    r_magnitude = torch.norm(r_prime, dim=1)  
    cross_product = I * torch.cross(dR_length * coil_vector, r_prime, dim=1) / (r_magnitude ** 3)[:,None]
    
    B = torch.sum(cross_product, dim=0) * (mu_iron / (4 * torch.pi)) 
    return B

def magnetic_falloff_adapter(rx_pos, tx_pos, *custom_args):
    
    tx_orientation = custom_args[0]        
        
    A = torch.zeros(3, rx_pos.shape[0], tx_pos.shape[0])
    
    for m in range(rx_pos.shape[0]):
        #pcnt = 100.0 * m / (rx_pos.shape[0] - 1) if rx_pos.shape[0] > 1 else 100.0
        #print('generating falloff matrix %.2f%%' % (pcnt,), end='\r')
        
        for n in range(tx_pos.shape[0]):
            r = tx_orientation[n] @ (rx_pos[m] - tx_pos[n])
            field_vector = magnetic_field_iron_bar_core(r, *custom_args[1:])
            A[:, m, n] = tx_orientation[n].T @ field_vector    
    #print('')
    return A
 
def delay_matrix_from_positions(rx_positions, tx_positions, sample_rate, signal_speed):
    rx_pos = rx_positions[:,None,:]
    tx_pos = tx_positions[None,:,:]
    d = torch.sqrt(torch.sum((tx_pos - rx_pos) ** 2, 2)) * sample_rate / signal_speed
    return d

#delay_fft
#if given an integer, returns the fft of a vector with 1.0 in that position
def delay_fft(delay, data_len, real=False, use_linear_resample=False):
    if real:
        fft_len = data_len // 2 + 1
    else:
        fft_len = data_len            
    v_delay = delay * -2.0 * torch.pi * torch.arange(fft_len) / data_len
    return torch.exp(1j * v_delay)

def fft_sim_transmit_1d_to_3d(n_tx, n_rx, n_data, sample_rate, signal_speed,
                               tx_pos, rx_pos, tx,
                               use_custom_falloff=None, custom_falloff_args=None):
    
    tx_fft = torch.fft.rfft(tx)

    delay = delay_matrix_from_positions(rx_pos, tx_pos, sample_rate, signal_speed)    
    falloff = use_custom_falloff(rx_pos, tx_pos, *custom_falloff_args)
            
    a_fft = delay_fft(delay[:,:,None], n_data, real=True)[None,:,:,:] * falloff[:,:,:,None]
    a_fft = torch.reshape(a_fft, (3*n_rx, n_tx, tx_fft.shape[-1]))
        
    _rx_fft = a_fft * tx_fft[None,:,:]
    _rx_fft = torch.sum(_rx_fft, dim=1)
    _rx_fft = _rx_fft.reshape(3, n_rx, -1)
    _rx = torch.fft.irfft(_rx_fft, n=tx.shape[-1])
    
    _rx = torch.transpose(_rx, 0, 1)
    _rx = torch.transpose(_rx, 1, 2)
    
    return _rx

#solve_mixed_transmission
#provide positions of transmitters and recievers
#provide the signal you would like each reciever to experience in rx
#returns the proper amplitude to transmit
def solve_mixed_transmission_1d_to_3d(n_tx, n_rx, n_data, sample_rate, signal_speed,
                               tx_pos, rx_pos, rx,
                               use_custom_falloff=None, custom_falloff_args=None):
    
    rx = torch.transpose(rx, 1, 2)
    rx = torch.transpose(rx, 0, 1)
    rx_fft = torch.fft.rfft(rx)

    delay = delay_matrix_from_positions(rx_pos, tx_pos, sample_rate, signal_speed)
    
    falloff = use_custom_falloff(rx_pos, tx_pos, *custom_falloff_args)
            
    # [1, n_rx, n_tx, fft_len] * [3, n_rx, n_tx, 1]
    a_fft = delay_fft(delay[:,:,None], n_data, real=True)[None,:,:,:] * falloff[:,:,:,None]
         
    a_fft = torch.reshape(a_fft, (3*n_rx, n_tx, rx_fft.shape[-1]))
    
    # pinv needs shape [fft_len, 3*n_rx, n_tx]
    # to broadcast pinv across fft_len
    inv_a_fft = torch.transpose(a_fft, 0, 2)
    inv_a_fft = torch.transpose(inv_a_fft, 1, 2)
    inv_a_fft = torch.linalg.pinv(inv_a_fft)
    inv_a_fft = torch.transpose(inv_a_fft, 0, 2)
    inv_a_fft = torch.transpose(inv_a_fft, 0, 1)
         
    #a broadcasted matmul [n_tx, n_rx*3] across fft_len    
    tx_fft = inv_a_fft * rx_fft.reshape(1, 3 * n_rx, -1)  
    tx_fft = torch.sum(tx_fft, dim=1)

    tx = torch.fft.irfft(tx_fft, n=rx.shape[-1])
    return tx

def simple_test():

    I = 150000
    mu_0 = 1.25663706212e-6    # Vacuum permeability  kg⋅m⋅s−2·A−2
    N = 200  # Number of turns in the coil
    L = 0.02  # Length of the iron bar core in meters
    R = 0.002 # Radius of the iron bar core in meters
    integration_steps = 10000  # number of integration steps
    mu_r_iron = 1000  # Relative permeability of iron bar

    sample_rate = 2400000
    signal_speed = 299792458    #speed of light m/sec

    n_tx = 6
    n_rx = 2
    n_data = 1  

    tx_pos = torch.rand(n_tx, 3) * 2.0 - 1.0
    rx_pos = torch.rand(n_rx, 3) * 2.0 - 1.0
    
    tx_orientation = torch.zeros(n_tx, 3, 3)
    tx_orientation[:,0,0] = 1
    tx_orientation[:,1,1] = 1
    tx_orientation[:,2,2] = 1

    rx = torch.zeros( (n_rx, n_data, 3) )
    rx[0,:,:] = 1.0

    magnet_falloff_args = (tx_orientation, I, mu_0, N, L, R, integration_steps, mu_r_iron)
    falloff = magnetic_falloff_adapter(rx_pos, tx_pos, *magnet_falloff_args)
    tx = solve_mixed_transmission_1d_to_3d(n_tx, n_rx, n_data, sample_rate, signal_speed, tx_pos, rx_pos, rx, 
                                  magnetic_falloff_adapter, magnet_falloff_args )
        
    print('falloff for all tx at reciever 0  falloff[:,0,:].T')
    print(falloff[:,0,:].T)
    print('power for all tx')
    print(tx)

    total = falloff[:,0,:].T * tx
    print('fields at reciever 0')
    print(total)
    print('sum of fields at reciever 0')
    print(torch.sum(total,dim=0))
    print('requested at receiver 0')
    print(rx[0])
    
    total = falloff[:,1,:].T * tx
    print('sum of fields at reciever 1')
    print(torch.sum(total,dim=0))
    print('requested at receiver 1')
    print(rx[1])


print('running simple test')
simple_test()
print('finished simple test\n')


I = 1.5  # Current in amperes
mu_0 = 1.25663706212e-6    # Vacuum permeability  kg⋅m⋅s−2·A−2
N = 200  # Number of turns in the coil
L = 0.02  # Length of the iron bar core in meters
R = 0.002 # Radius of the iron bar core in meters
integration_steps = 10000  # number of integration steps
mu_r_iron = 1000  # Relative permeability of iron bar

n_tx = 24
n_rx = 8
n_data = 1                  #prefer odd size fft, DC=1

sample_rate = 2400000
signal_speed = 299792458    #speed of light m/sec

#choose any positions, and any desired 3d magnetic field to be recieved at rx_pos

tx_pos = torch.rand(n_tx, 3) * 2.0 - 1.0
rx_pos = torch.rand(n_rx, 3) * 2.0 - 1.0

tx_orientation = torch.rand(n_tx, 3, 3)
tx_orientation, _ = torch.linalg.qr(tx_orientation)      #random orthogonal matrices

rx = torch.rand( (n_rx, n_data, 3) ) * 2.0 - 1.0

magnet_falloff_args = (tx_orientation, I, mu_0, N, L, R, integration_steps, mu_r_iron)

tx = solve_mixed_transmission_1d_to_3d(n_tx, n_rx, n_data, sample_rate, signal_speed, tx_pos, rx_pos, rx, 
                              magnetic_falloff_adapter, magnet_falloff_args )

# transmit at I * tx from tx_pos to recieve rx at rx_pos

_rx = fft_sim_transmit_1d_to_3d(n_tx, n_rx, n_data, sample_rate, signal_speed, tx_pos, rx_pos, tx, 
                              magnetic_falloff_adapter, magnet_falloff_args )

abs_error = torch.abs(rx - _rx)
print('predict iron bar magnet amplitude for desired magnetic field')
print('field max abs error', torch.amax(abs_error))
print('field mean abs error', torch.mean(abs_error))
print('field min abs error', torch.amin(abs_error))

# possibly could collapse rx back to an amplitude, and get 1:1 instead of 3:1, in a random direction
# sqrt( r0^2 + r1^2 + r2^2 ), not sure though
