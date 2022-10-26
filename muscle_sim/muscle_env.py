import numpy as np

class Three_element_env(object):
    """
    pneumatic muscle simulator class based on three element model

    Attributes:
        t_sample: sampling time
        t_global: global time 
        mass: inertia mass of the muscle
        load_0: initial external load
        load: external load in k squence
        state: state vector of muscle [pos(k),pos(k-1)]^T
        pressure: pressure input of the muscle
        contraction: contraction state
        f: force factor
        k: spring factor
        b: damping factor
    """
    def __init__(self,T_sample,Load_0,Mass) :
        self.t_sample = T_sample
        self.load_0 = Load_0
        self.state = np.zeros(2, dtype=float)
        self.mass = Mass
        self.reset()

    def reset(self):
        """
        reset the simulator to initial state
        """
        self.load = self.load_0
        self.t_global = 0.0
        self.state[0] = self.state[1] = 0
        self.contraction = True

    def step(self, U_k,Load_k):
        """
        update the system state by one step of sampling time

        Args:
            U_k: pressure input
            Load_k: external load

        Returns:
            self.state: updated state vector 
        """
        self.f, self.k,self.b = self.calculate_coe(
            pressure = U_k,
            contraction= self.contraction
        )
        self.pressure = U_k
        self.load = Load_k
        pos_0 = self.state[1] # pos(k-1)
        pos_1 = self.state[0] # pos(k)

        #pos(k+1)
        pos_2 = (self.t_sample **2 ) * (self.f - self.load) - (self.k*(self.t_sample**2)-2*self.mass-self.b*self.t_sample)*pos_1 -self.mass*pos_0
        pos_2 = pos_2 / (self.mass + self.b * self.t_sample)

        # k <- k+1
        # update state vector
        self.state[0] = pos_2
        self.state[1] = pos_1

        # update contraction
        if(pos_2 - pos_1 >=0):
            self.contraction = True
        else:
            self.contraction = False

        # update global time
        self.t_global += self.t_sample

        return self.state

    def get_t(self):
        """
        acquire the current global time of the simulation

        returns:
            self.t_global: current global time 
        """
        return self.t_global

    def render(self):
        """
            illustrate the current state of the simulation
        """
        pass

    def calculate_coe(self,pressure:float, contraction:bool) :
        """
        calculate model parameters

        Args:
            pressure: muscle pressure input
            contraction: contraction state of the muscle

        Returns:
            [
            f: Force coeficient,
            k: Spring coeficient,
            b: Damping coeficient
            ]

        """

        if pressure >= 550:
            pressure = 550
        elif pressure <= 150:
            pressure = 150
        
        if 150 <= pressure <= 314:
            k = 32.7 - 0.0321 * pressure
        elif 314 < pressure <= 550:
            k = 17 + 0.0179 * pressure
        
        f = 2.91 * pressure + 44.6

        if contraction:
            b = 2.9
        else:
            if 150 <= pressure <= 372:
                b = 1.57
            elif 372 <= pressure <= 550:
                b = 0.311 + 0.00338 * pressure
        return f, k, b
    