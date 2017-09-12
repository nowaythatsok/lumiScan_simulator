from oFuncs import *

c = container1(1000,2**3,10000)

c.roll_the_dice()

c.plot_percentage()

c = container1(1000,2**6,10000)

c.roll_the_dice()

c.plot_percentage()

c = container1(1000,2**9,10000)

c.roll_the_dice()

c.plot_percentage()

c = container1(1000,2**12,10000)

c.roll_the_dice()

c.plot_percentage()

plt.show()
