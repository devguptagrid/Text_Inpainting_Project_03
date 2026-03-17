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


def plot_confident_mistakes(mistakes_per_step, total_tokens_per_step):
    """
    Plot confident mistake rate over diffusion steps
    """


    steps = list(range(len(mistakes_per_step)))

    # compute rate
    rates = []
    for m, t in zip(mistakes_per_step, total_tokens_per_step):
        rate = m / t if t > 0 else 0
        rates.append(rate)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(steps, rates)
    plt.xlabel("Diffusion Step")
    plt.ylabel("Confident Mistake Rate")
    plt.title("Confident Mistakes vs Diffusion Step")
    plt.show()

