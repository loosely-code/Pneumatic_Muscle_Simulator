from matplotlib.pyplot import plot
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

    def get_t(self):
        """
        acquire the current global time of the signal

        returns:
            self.t_global: current global time 
        """
        return self.t_global

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

# test 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # define the parameters
    t_total = 5.0  # 5s in realtime
    t_sample = 0.01
    N_iter = int(t_total // t_sample) 

    # initialization

    signal = signal_sin(
        Amp= 1,
        Period= 2.5,
        Center= 0,
        T_sample=t_sample
    )

    plotMemory_output = np.zeros([N_iter, 1])
    plotMemory_time = np.zeros([N_iter, 1])

    for iter in range(N_iter):
        plotMemory_output[iter] = signal.step()
        plotMemory_time[iter] = signal.get_t()

    #plot

    #fig =plt.figure()
    #a=fig.add_axes([0,0,5,1])
    #a.plot(plotMemory_time,plotMemory_output)
    # plot
    fig, a = plt.subplots(1, 1)
    a.set_title("signal output")
    a.set_xlabel("t - s")
    a.set_ylabel("y - mm")
    a.plot(plotMemory_time, plotMemory_output)
    a.grid(True)
    plt.show()
