import matplotlib.pyplot as plt

def plot_metrics(avg_confidence, avg_entropy):
    """
    Plot confidence and entropy over diffusion steps
    """


    steps = list(range(len(avg_confidence)))

    # Plot Confidence
    plt.figure()
    plt.plot(steps, avg_confidence)
    plt.xlabel("Diffusion Step")
    plt.ylabel("Average Confidence")
    plt.title("Confidence vs Diffusion Step")
    plt.show()

    # Plot Entropy
    plt.figure()
    plt.plot(steps, avg_entropy)
    plt.xlabel("Diffusion Step")
    plt.ylabel("Average Entropy")
    plt.title("Entropy vs Diffusion Step")
    plt.show()

