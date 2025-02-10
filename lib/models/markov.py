import numpy as np
import matplotlib.pyplot as plt

class Markov():
    '''
    :param init_state: the initial state of the system, as a probability vector 
    :param trans_mat: a square matrix representing probabilities of moving from one state to another
    '''
    def __init__(self, init_state: np.array, trans_mat: np.array):
        self.init_state = init_state
        self.trans_mat = trans_mat
    
    def compute_state(self, iteration: int):
        return np.matmul(np.linalg.matrix_power(self.trans_mat, iteration), self.init_state)

    def plot_states(self, num_iter: int):
        iterations = np.zeros((num_iter + 1, len(self.init_state)))
        out = self.init_state
        for i in range(1, num_iter + 1):
            out = np.matmul(self.trans_mat, out)
            iterations[i] = out
            print(f'out({i}) = {out}')
        fig, axes = plt.subplots(1, len(self.init_state), figsize=(len(self.init_state) * 5, 5))
        fig.suptitle('Markov Chain iterations')

        if len(self.init_state) == 1:
            axes = [axes]

        for i in range(len(self.init_state)):
            axes[i].plot(range(num_iter + 1), iterations[:, i], 'o-')
            axes[i].set_title(f"State {i + 1} over iterations")
            axes[i].set_xlabel("Iteration")
            axes[i].set_ylabel("State Value")
        
        plt.tight_layout()
        plt.show()

    def get_steady_state(self):
        mat = self.trans_mat - np.eye(self.trans_mat.shape[1])
        # sum of steady state probabilities must be 1
        for i in range(mat.shape[1]):
            mat[0,i] = 1
        p0 = np.zeros((mat.shape[1], 1))    
        p0[0] = 1
        # take inverse to solve for steady state
        return np.matmul(np.linalg.inv(mat), p0) 


# example (note: Markov chain sol will always converge to a steady state solution, so it's a bad estimation after a long time for our systems)
A = np.array(
    [[.5,.2,.3],
     [.3,.8,.3],
     [.2, 0,.4]])

x_0 = np.array([0.60, 0.2, 0.2])
markov = Markov(x_0, A)

print(markov.compute_state(5))
print(markov.get_steady_state())
markov.plot_states(10)
