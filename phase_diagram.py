import matplotlib.pyplot as plt

def draw_phase_diagram(x, y):
	plt.scatter(x, y)
	plt.title("Phase diagram")
	plt.xlabel("y")
	plt.ylabel("x")
	plt.show()
