import numpy as np

class Three_element_env(object):
    def __init__(self,T_sample,load_0) -> None:
        self.T_sample = T_sample
        self.load_0 = load_0
        self.state = np.zeros(2, dtype=float)
        self.reset()

    def reset(self):
        self.load = self.load_0
        self.t_global = 0.0
        self.state[0] = self.state[1] = 0
        self.contraction = True

    def step(self, u_k,load_k):
        pass

    def calculate_coe(pressure,contraction):
        k = 0.0
        if 150 <= pressure <= 314:
            k = 32.7 - 0.0321 * pressure
        elif 314 < pressure <= 550:
            k = 17 + 0.0179 * pressure
        f = 2.91 * pressure + 44.6
        b = 0.0
        if contraction:
            b = 2.9
        else:
            if 150 <= pressure <= 372:
                b = 1.57
            elif 372 <= pressure <= 550:
                b = 0.311 + 0.00338 * pressure
        return f, k, b
    
