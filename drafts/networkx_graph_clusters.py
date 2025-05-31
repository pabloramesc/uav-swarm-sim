import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Matriz de conexión (adyacencia binaria)
adj_matrix = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0]
])

# Crear grafo desde matriz de adyacencia
G = nx.from_numpy_array(adj_matrix)

# Obtener componentes conexas (clústeres)
clusters = list(nx.connected_components(G))

# Asignar un color distinto a cada componente
color_map = {}
for i, cluster in enumerate(clusters):
    for node in cluster:
        color_map[node] = i

# Crear lista de colores en orden de nodos
colors = [color_map[node] for node in G.nodes]

# Dibujar grafo
pos = nx.spring_layout(G, seed=42)  # O usa nx.kamada_kawai_layout(G) si prefieres
nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.Set1, node_size=800, font_color='white')
plt.title("Clústeres de conectividad de la red de drones")
plt.show()
