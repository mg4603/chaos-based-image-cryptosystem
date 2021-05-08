from sine_map import SineMap
import numpy as np
import matplotlib.pyplot as plt
amin = 0
amax = 3
mult_a = (amax - amin) /200000

bmin = 0
bmax = 3
mult_b = (bmax - bmin) / 200000


avalues = np.arange(amin, amax, 0.0001)
bvalues = np.arange(bmin, bmax, 0.0001)


maps_ax = []
maps_ay = []
maps_a = []
for a in avalues:
    x = np.array((0.1, 0.1))
    b = 2.6
    result = []
    SINE_MAP = SineMap(a, b)
    for t in range(100):
    	x = SINE_MAP.f(x)

    for t in range(20):
        x = SINE_MAP.f(x)
        maps_a.append(x)    


maps_bx = []
maps_by = []
maps_b = []
for b in bvalues:
    x = np.array((0.1, 0.1))
    a = 0.95
    result = []
    SINE_MAP = SineMap(a, b)
    for t in range(100):
    	x = SINE_MAP.f(x)

    for t in range(20):
        x = SINE_MAP.f(x)
        maps_b.append(x)

for i in range(len(maps_a)):
	maps_ax.append(maps_a[i][0])
	maps_ay.append(maps_a[i][1])
	maps_bx.append(maps_b[i][0])
	maps_by.append(maps_b[i][1])


"x sequence a"
aticks = np.arange(amin, amax, (1/200000))
print(type(aticks))
print(type(maps_ax))
print(aticks.shape)
print(len(maps_ax))
print(aticks)
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(aticks, maps_ax, 'b.',alpha = 1)
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, 1)
ax1.set_xlabel('a')
ax1.set_ylabel('x')
ax1.legend(loc='best')
ax1.set_title('Bifurcation Diagram')
plt.show()


"y sequence a"
fig1 = plt.figure(figsize=(10,7))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(aticks, maps_ay, 'b.',alpha = 1)
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, 1)
ax1.set_xlabel('a')
ax1.set_ylabel('y')
ax1.legend(loc='best')
ax1.set_title('Bifurcation Diagram')
plt.show()


bticks = np.arange(bmin, bmax, (1/200000))
"x sequence b"
fig2 = plt.figure(figsize=(10,7))
ax1 = fig2.add_subplot(1,1,1)
ax1.plot(bticks, maps_bx, 'b.',alpha = 1)
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, 1)
ax1.set_xlabel('b')
ax1.set_ylabel('x')
ax1.legend(loc='best')
ax1.set_title('Bifurcation Diagram')
plt.show()


"y sequence b"
fig3 = plt.figure(figsize=(10,7))
ax1 = fig3.add_subplot(1,1,1)
ax1.plot(bticks, maps_by, 'b.',alpha = 1)
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, 1)
ax1.set_xlabel('b')
ax1.set_ylabel('y')
ax1.legend(loc='best')
ax1.set_title('Bifurcation Diagram')
plt.show()
