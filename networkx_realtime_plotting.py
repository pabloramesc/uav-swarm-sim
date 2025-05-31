import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class ConnectionGraphPlotter:
    def __init__(self, num_nodes, positions=None):
        """
        Inicializa el visualizador del grafo de conexión.
        
        Parámetros:
        - num_nodes: número total de drones
        - positions: diccionario opcional {i: (x, y)} con las posiciones físicas
        """
        self.num_nodes = num_nodes
        self.positions = positions
        self.fig, self.ax = plt.subplots()
        plt.ion()  # Modo interactivo
        self.fig.show()
        self.fig.canvas.draw()

    def update(self, adjacency_matrix, positions=None):
        """
        Actualiza el grafo con una nueva matriz de conexión y opcionalmente nuevas posiciones.
        
        Parámetros:
        - adjacency_matrix: matriz binaria NxN indicando conectividad
        - positions: diccionario opcional {i: (x, y)} para actualizar posiciones
        """
        G = nx.from_numpy_array(np.array(adjacency_matrix))
        
        # Calcular componentes conexas
        clusters = list(nx.connected_components(G))
        color_map = {}
        for i, cluster in enumerate(clusters):
            for node in cluster:
                color_map[node] = i
        colors = [color_map[node] for node in G.nodes]

        # Actualizar posiciones si se proporcionan
        if positions:
            self.positions = positions
        elif not self.positions:
            self.positions = nx.spring_layout(G, seed=42)

        # Limpiar el gráfico anterior
        self.ax.clear()
        nx.draw(
            G,
            pos=self.positions,
            ax=self.ax,
            with_labels=True,
            node_color=colors,
            cmap=plt.cm.Set1,
            node_size=800,
            font_color='white'
        )
        self.ax.set_title("Ad-hoc network topology")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    import time

    # Inicializar
    plotter = ConnectionGraphPlotter(num_nodes=5)

    # Lista de matrices de conexión para simular distintas fases
    matrices = [
        # Todos aislados
        np.eye(5, dtype=int),

        # Dos pares conectados, uno solo
        np.array([
            [1,1,0,0,0],
            [1,1,0,0,0],
            [0,0,1,1,0],
            [0,0,1,1,0],
            [0,0,0,0,1],
        ]),

        # Red parcialmente conectada en dos grupos
        np.array([
            [1,1,1,0,0],
            [1,1,1,0,0],
            [1,1,1,0,0],
            [0,0,0,1,1],
            [0,0,0,1,1],
        ]),

        # Conectividad total
        np.ones((5,5), dtype=int),

        # Desconexión de uno de los nodos
        np.array([
            [1,1,1,1,0],
            [1,1,1,1,0],
            [1,1,1,1,0],
            [1,1,1,1,0],
            [0,0,0,0,1],
        ]),

        # Conectividad en anillo
        np.array([
            [1,1,0,0,1],
            [1,1,1,0,0],
            [0,1,1,1,0],
            [0,0,1,1,1],
            [1,0,0,1,1],
        ]),
    ]

    for matrix in matrices:
        plotter.update(matrix)
        time.sleep(2.0)