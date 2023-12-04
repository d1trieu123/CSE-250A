import numpy as np
import matplotlib.pyplot as plt
import string

def load_data(observations_fh, transitionMtx_fh, emissionMtx_fh, initialState_fh):
    observations = np.loadtxt(observations_fh, dtype='int')  
    transition = np.loadtxt(transitionMtx_fh)  
    emission = np.loadtxt(emissionMtx_fh)  
    initialState = np.loadtxt(initialState_fh)  
    return observations, transition, emission, initialState

def initialize_variables(n, T, initialState, emission, observations):
    l = np.empty([n, T])
    l[:, 0] = np.log(initialState) + np.log(emission[:, observations[0]])

    phi = np.empty([n, T])
    phi[:, 0] = initialState

    s = np.full(T, -1, dtype=int)

    return l, phi, s

def update(row, col, l, transition, observations, emission):
    state_transitions = l[:, col - 1] + np.log(transition[:, row])
    most_likely = np.argmax(state_transitions)
    new_l = np.amax(state_transitions) + np.log(emission[row, observations[col]])
    return most_likely, new_l

def backtrack(t_idx, T, l, transition, s):
    if t_idx == T - 1:
        return np.argmax(l[:, T - 1])
    else:
        return np.argmax(l[:, t_idx] + np.log(transition[:, s[t_idx + 1]]))

def run_viterbi(T, n, l, phi, update, backtrack, transition, observations, emission, s):
    for t in range(T - 1):
        for j in range(n):
            phi[j, t + 1], l[j, t + 1] = update(j, t + 1, l, transition, observations, emission)

    for t in range(T - 1, -1, -1):
        s[t] = backtrack(t, T, l, transition, s)

def plot_hmm(s):
    plt.plot(s)
    plt.title('State vs Time')
    plt.xlabel('Time ')
    plt.ylabel('Hidden States')
    plt.show()

def decode(alpha_dict, s, T):
    message = [alpha_dict.get(s[0] + 1)]
    for t in range(1, T):
        if s[t] != s[t - 1]:
            message.append(alpha_dict.get(s[t] + 1))
    return ''.join(message)

def main():
    observations_fh = 'HW7/observations.txt'
    transitionMtx_fh = 'HW7/transitionMatrix.txt'
    emissionMtx_fh = 'HW7/emissionMatrix.txt'
    initialState_fh = 'HW7/initialStateDistribution.txt'

    observations, transition, emission, initialState = load_data(
        observations_fh, transitionMtx_fh, emissionMtx_fh, initialState_fh
    )

    n = 27
    T = 430000
    alpha_dict = dict(zip(range(1, 28), string.ascii_lowercase + ' '))

    l, phi, s = initialize_variables(n, T, initialState, emission, observations)

    run_viterbi(T, n, l, phi, update, backtrack, transition, observations, emission, s)

    print(decode(alpha_dict, s, T))
    plot_hmm(s)

if __name__ == "__main__":
    main()