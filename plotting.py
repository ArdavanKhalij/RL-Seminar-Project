import matplotlib.pyplot as plt

def plot_lists(list1, list2, title, x_label, y_label, legend_labels, each):
    LIST1 = []
    LIST2 = []
    x1 = 0
    x2 = 0
    for i in range(0, len(list1)):
        x1 = x1 + list1[i]
        x2 = x2 + list2[i]
        if i % each == 0:
            LIST1.append(x1/each)
            LIST2.append(x2 / each)
            x1 = 0
            x2 = 0

    plt.plot(LIST1, label=legend_labels[0], color='blue')
    plt.plot(LIST2, label=legend_labels[1], color='red')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()