import numpy as np

class signal_sin:
    """
    generate sin signal 

    Attributes:
        amp:
        period:
        center:
        w:
        t_sample:
        t_global:
        output:

    """
    def __init__(self, Amp, Period, Center, T_sample ):
        self.amp = Amp
        self.period = Period
        self.center = Center
        self.w = (2 * np.pi) / self.period
        self.t_sample = T_sample
        self.reset()

    def reset(self):
        """
        reset the signal to initial state        
        """
        self.t_global = 0
        self.output = 0

    def step(self):
        """
        update the signal by one step of sampling time

        Args:

        Returns:
            self.output: updated signal output
        """
        self.t_global += self.t_sample
        self.output = self.center + self.amp * (np.sin(self.w * self.t_global))
        return self.output

    def search(self,T_global):
        """
        search the signal value from a given global time

        Args:
            T_global: global time

        Returns:
            result: signal value of the given global time
        """
        result = self.center + self.amp * (np.sin(self.w * T_global))
        return result

