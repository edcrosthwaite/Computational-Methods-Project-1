import matplotlib.pyplot as plt

def plot(data, y=(0,0), x=(2, 0)):
    """Creates plot
    """
    for i in range(3):
        plt.cla()

        plt.plot(data[x], data[1, i])       # Sprung position
        plt.plot(data[x], data[0 ,i])       # Unsprung position

        # Automatilcally scales x and y axis depending on data
        plt.autoscale(True)
        plt.xlim(left=0)
        plt.legend(["Sprung", "Unsprung"])
        plt.grid(True)

        plt.savefig(str(i))

    plt.cla()
    plt.plot(data[x], data[2,1])    # Bump input
    plt.savefig("Bump")
