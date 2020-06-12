import matplotlib
matplotlib.use('tkagg') # set the backend to tk, using agg renderer
import matplotlib.pyplot as plt
plt.ion()
plt.figure(1)
plt.plot([1]*10)