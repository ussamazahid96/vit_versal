import numpy as np
import matplotlib.pyplot as plt
timeaxis = np.linspace(0, 40, 400)
po = np.loadtxt("vck190_power.log")
timeaxis = timeaxis[15*10 : 30*10]
po = po[15*10 : 30*10]
plt.plot(timeaxis, po, 'b-')
plt.grid()
plt.savefig("./power_plot_vck190.png")
