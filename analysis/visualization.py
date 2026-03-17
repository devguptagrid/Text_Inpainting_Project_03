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


def plot_confidence_histogram(step_counts, total_tokens):
    """
    Plot percentage of tokens reaching confidence > 0.9 at each step
    """

    
    import matplotlib.pyplot as plt

    steps = list(step_counts.keys())

    # convert to percentage
    percentages = [
        (step_counts[s] / total_tokens) * 100 if total_tokens > 0 else 0
        for s in steps
    ]

    plt.figure()
    plt.bar(steps, percentages)
    plt.xlabel("Step where confidence > 0.9")
    plt.ylabel("Percentage of Tokens (%)")
    plt.title("Confidence Histogram (Percentage)")
    plt.show()
    

def plot_entropy_correct_vs_incorrect(entropy_correct, entropy_incorrect):
    """
    Plot entropy comparison for correct vs incorrect tokens
    """

    import matplotlib.pyplot as plt

    steps = list(range(len(entropy_correct)))

    plt.figure()
    plt.plot(steps, entropy_correct, label="Correct Tokens")
    plt.plot(steps, entropy_incorrect, label="Incorrect Tokens")

    plt.xlabel("Diffusion Step")
    plt.ylabel("Average Entropy")
    plt.title("Entropy: Correct vs Incorrect Tokens")
    plt.legend()
    plt.show()
    


def plot_entropy_heatmaps(entropy_correct, entropy_incorrect):
    """
    Plot entropy heatmaps for correct and incorrect tokens
    """

    
    import matplotlib.pyplot as plt

    # Correct tokens heatmap
    if entropy_correct.numel() > 0:
        plt.figure()
        plt.imshow(entropy_correct, aspect='auto')
        plt.colorbar()
        plt.xlabel("Diffusion Step")
        plt.ylabel("Token Index")
        plt.title("Entropy Heatmap (Correct Tokens)")
        plt.show()

    # Incorrect tokens heatmap
    if entropy_incorrect.numel() > 0:
        plt.figure()
        plt.imshow(entropy_incorrect, aspect='auto')
        plt.colorbar()
        plt.xlabel("Diffusion Step")
        plt.ylabel("Token Index")
        plt.title("Entropy Heatmap (Incorrect Tokens)")
        plt.show()
    

