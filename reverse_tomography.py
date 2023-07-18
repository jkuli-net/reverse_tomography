import torch

def shift_signal(signal, t):
    f_t = t - torch.floor(t)
    i_t = int(torch.floor(t))
    #i could sum two ranges instead of two rolls, clamping to valid ranges
    return torch.roll(signal, i_t, 0) * (1-f_t) + torch.roll(signal, i_t+1, 0) * f_t

def sim_recieve(num_transmit, num_receive, signal_length, transmit, transmit_positions, receive_positions,
               signal_speed, sample_rate):
    
    recieve = torch.zeros( (num_receive, signal_length) , device=transmit.device)
    
    #ill assume for now the initial volume was measured at distance 1
    falloff_per_double = 0.5011872336272722 # -6db
        
    for irecv in range(num_receive):
        for itran in range(num_transmit):            
            distance = torch.sqrt(torch.sum((receive_positions[irecv] - transmit_positions[itran]) ** 2))
            delay = distance * sample_rate / signal_speed            
            falloff = falloff_per_double ** torch.log2(distance / 1)
                                    
            recieve[irecv] += shift_signal(transmit[itran], delay) * falloff
            
    return recieve


sample_rate = 44100
signal_speed = 343.3

#8 speakers is sufficient to get 8 totally different signals at 8 specific positions
num_transmit = 16
num_receive = 8

signal_length = 1024

receive_positions = torch.rand(num_receive, 3) 
transmit_positions = torch.rand(num_transmit, 3) 

transmit = torch.zeros( (num_transmit, signal_length) )
receive_truth = torch.rand( (num_receive, signal_length) )

if 0:
    #i was able to get silence on 7 of 8, but it took 18 transmitters to get it accurate
    receive_truth[1:] = 0

if torch.cuda.is_available():
    device = torch.device("cuda")
    receive_positions = receive_positions.to(device)
    transmit_positions = transmit_positions.to(device)
    transmit = transmit.to(device)
    receive_truth = receive_truth.to(device)

transmit.requires_grad = True

#steps and learning rate will need adjusted, especially for harder problems
lr = 0.1
steps = 256
optimizer = torch.optim.Adam([transmit],  lr=lr)
mse_loss = torch.nn.MSELoss()
for step in range(steps):
    recieve_pred = sim_recieve(num_transmit, num_receive, signal_length, transmit,
                          transmit_positions, receive_positions, signal_speed, sample_rate)
    loss = mse_loss(recieve_pred, receive_truth)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('step %i  loss %0.15f\r' % (step, loss), end='')
print('')


print('predict audio to minimize(recieve_pred - receive_truth)')
print( 'recieve_pred max error', torch.amax(torch.abs(recieve_pred - receive_truth)).item() )
print( 'recieve_pred mean error', torch.mean(torch.abs(recieve_pred - receive_truth)).item() )
print( 'recieve_pred min error', torch.amin(torch.abs(recieve_pred - receive_truth)).item() )
