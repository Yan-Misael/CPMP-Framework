from generation.adapters.input.input_adapter import InputAdapter
import numpy as np
from cpmp.layout import Layout

class EnrichedLayoutAdapter(InputAdapter):
    stack_adapter = None
    extra_data_adapter = None

    def __init__(self, layout_adapter, stack_features_adapter):
        super().__init__({
            "S": np.float32,
            "X": np.float32
        })
        self.layout_adapter = layout_adapter()
        self.stack_features_adapter = stack_features_adapter()

    def input_2_vec(self, layout: Layout, H: int):
        S = self.layout_adapter.input_2_vec(layout, H)[0]
        X = self.stack_features_adapter.to_vec(layout, H)
        return S, X

    def add(self, layout_data):
        S_matrix, X = layout_data

        self.data['S'].append(S_matrix)
        self.data['X'].append(X)