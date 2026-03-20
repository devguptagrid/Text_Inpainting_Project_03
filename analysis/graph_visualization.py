import networkx as nx
import matplotlib.pyplot as plt

def plot_transition_graph(tokens, probs, title="Transition Graph"):
    """
    Create directed graph of token transitions
    """

    G = nx.DiGraph()

    source = "[MASK]"

    # add nodes and edges
    for tok, p in zip(tokens, probs):
        tok = tok.replace("##", "")  # clean subwords
        G.add_edge(source, tok, weight=p.item())

    pos = nx.spring_layout(G, seed=42)

    # draw nodes
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10)

    # draw edge labels (probabilities)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(title)
    plt.show()
    
