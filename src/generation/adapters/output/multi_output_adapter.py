from generation.adapters.output.output_adapter import OutputAdapter
from generation.adapters.output.action_adapter import ActionAdapter
from generation.adapters.output.cost_adapter import CostAdapter
import numpy as np

class MultiOutputAdapter(OutputAdapter):
    def __init__(self):
        super().__init__({
            "Y": np.int32,
            "cost": np.float32
        })
        self.action_adapter = ActionAdapter()
        self.cost_adapter = CostAdapter()
    
    def output_2_vec(self, moves, S, cost):
        Y = self.action_adapter.output_2_vec(moves, S, cost)
        cost = self.cost_adapter.output_2_vec(moves, S, cost)
        return Y, cost
    
    def add(self, output_data):
        Y, cost = output_data
        self.data['Y'].append(Y)
        self.data['cost'].append(cost)