from generation.adapters.input.stack_features.stack_features_adapter import StackFeaturesAdapter
from cpmp.layout import Layout
import numpy as np

class StackFeatures3FAdapter(StackFeaturesAdapter):
    def to_vec(self, layout: Layout, H: int):
        X = np.zeros((len(layout.stacks), 3), dtype=np.float32)

        for i in range(len(layout.stacks)):
            X[i][0] = 1.0 if layout.is_sorted_stack(i) else 0.0
            X[i][1] = len(layout.stacks[i]) / H
            X[i][2] = (layout.sorted_elements[i] / len(layout.stacks[i])) if len(layout.stacks[i]) != 0.0 else 1

        return np.array(X, dtype=np.float32)