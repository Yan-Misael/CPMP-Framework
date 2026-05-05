import numpy as np
from generation.adapters.input.input_adapter import InputAdapter

class GPIAdapter(InputAdapter):
    def __init__(self):
        super().__init__({
            "G": np.int32,
            "P": np.int32,
            "I": np.int32,
            "S": np.int32,
            "H": np.int32, 
        })

    def input_2_vec(self, layout, H):
        G = [] # Valores de grupo
        P = [] # Dónde se ubica el contenedor en su respectiva pila
        I = [] # En qué pila se encuentra el contenedor
        S = len(layout.stacks) # Número de pilas

        for i in range(S):
            for j in range(len(layout.stacks[i])):
                G.append(layout.stacks[i][j])
                P.append(j)
                I.append(i)

        return np.array(G), np.array(P), np.array(I), S, H
    
    def add(self, layout_data):
        G, P, I, S, H = layout_data

        self.data['G'].append(G)
        self.data['P'].append(P)
        self.data['I'].append(I)
        self.data['S'].append(S)
        self.data['H'].append(H)