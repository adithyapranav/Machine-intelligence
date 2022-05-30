import numpy as np
def reversal(lst):
    return [elements for elements in reversed(lst)]

class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        # TODO
        len_of_seq=len(seq)
        t=np.zeros((len_of_seq,self.N),dtype=int)
        nu=np.zeros((len_of_seq,self.N))
        m=int(0);
        while(m<self.N):
            nu[0,m]=self.B[m,self.emissions_dict[seq[0]]]*self.pi[m]
            m+=1

        for i in range(1,len_of_seq):
            for j in range(0,self.N):
                max_of_t=-1
                max_of_nu=-1
                for k in range(0,self.N):
                    loc_nu=self.A[k,j]*self.B[j,self.emissions_dict[seq[i]]]*nu[i-1,k]
                    if (max_of_nu<loc_nu):
                        max_of_t=k
                        max_of_nu=loc_nu                        
                t[i,j]=max_of_t
                nu[i,j]=max_of_nu
              
        max_of_t=-1
        max_of_nu=-1
        m=int(0)
        while(m<self.N):
            loc_nu=nu[len_of_seq-1,m]
            if (max_of_nu<loc_nu):
                max_of_t=m
                max_of_nu=loc_nu    
            m+=1            
        h_s_s=[max_of_t]
        for n in range(len_of_seq-1,0,-1):
            h_s_s.append(t[n,h_s_s[-1]])
        h_s_s=reversal(h_s_s)
        self.states_dict={value:key for key,value in self.states_dict.items()}  
        hidden_states_sequence=[self.states_dict[n] for n in h_s_s]
        return hidden_states_sequence
        
        pass