import torch
import h5py
from torch.utils.data import Dataset
from settings import DATA_FOLDER
import os
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, filepath, max_size=None):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.file = None

        with h5py.File(self.filepath, "r") as f:
            self.input_keys = list(f['input'].attrs['key_order'])
            self.output_keys = list(f['output'].attrs['key_order'])
            
            total_len = len(f['input'][self.input_keys[0]])
            self.dataset_len = total_len if max_size is None else min(total_len, max_size)

    def _open_file(self):
        self.file = h5py.File(self.filepath, "r")
        self.input_datasets = {k: self.file[f'input/{k}'] for k in self.input_keys}
        self.output_datasets = {k: self.file[f'output/{k}'] for k in self.output_keys}
        self.cost_dataset = self.file['C']
        
    def _to_tensor(self, val):
        """Helper para convertir datos a tensores de forma eficiente"""
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val)
        return torch.tensor(val)

    def __getitem__(self, idx):
        if self.file is None: 
            self._open_file()
            
        inputs = [self._to_tensor(self.input_datasets[k][idx]) for k in self.input_keys]
        outputs = [self._to_tensor(self.output_datasets[k][idx]) for k in self.output_keys]
        return tuple(inputs), tuple(outputs)
    
    def __len__(self):
        return self.dataset_len

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        state['input_datasets'] = None
        state['output_datasets'] = None
        state['cost_dataset'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.file = None

def load_dataset(filepath, max_size=None, verbose=True):
    dataset = H5Dataset(DATA_FOLDER / filepath, max_size)
    if verbose:
        print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset

def load_data_from_path(filepath):
    """Carga datos respetando la estructura de grupos input/output"""
    with h5py.File(filepath, "r") as f:
        # Extraemos keys de los grupos
        input_keys = list(f['input'].attrs['key_order'])
        output_keys = list(f['output'].attrs['key_order'])
        
        data = {}
        # Cargamos los inputs
        for k in input_keys:
            data[f'input/{k}'] = f['input'][k][:]
            
        # Cargamos los outputs
        for k in output_keys:
            data[f'output/{k}'] = f['output'][k][:]
            
        # El costo 'C' suele estar en la raíz según tu implementación anterior
        data['C'] = f['C'][:]
        
        # Guardamos metadatos de orden para la reconstrucción
        data['_input_order'] = input_keys
        data['_output_order'] = output_keys
        return data
    
def load_data(filename):
    return load_data_from_path(DATA_FOLDER / filename)

def generate_dataset(data_files, output_name, min_cost=0, max_cost=999999, max_size=999999):
    output_path = DATA_FOLDER / output_name
    all_data = {}
    input_order = []
    output_order = []
    
    for data_file in data_files:
        path = DATA_FOLDER / data_file
        if path.exists():
            data = load_data_from_path(path)
            if not all_data:
                # Inicializamos las listas para cada path completo (ej: 'input/layout')
                all_data = {k: [] for k in data.keys()}
                input_order = data['_input_order']
                output_order = data['_output_order']
            
            for k in data:
                all_data[k].append(data[k])

    if not all_data: return

    # Combinar todos los archivos leídos
    combined_data = {k: np.concatenate(all_data[k], axis=0) for k in all_data if not k.startswith('_')}
    
    # Crear máscara de filtrado por costo
    mask = (combined_data['C'] >= min_cost) & (combined_data['C'] <= max_cost)
    total_available = np.sum(mask)
    final_len = min(total_available, max_size)
    
    # Aplicar máscara y límite de tamaño
    for k in combined_data:
        combined_data[k] = combined_data[k][mask][:final_len]

    # Escritura del archivo con la nueva estructura
    with h5py.File(output_path, "w") as f:
        # 1. Crear Grupos
        g_input = f.create_group("input")
        g_output = f.create_group("output")
        
        # 2. Guardar Inputs
        for k in input_order:
            full_key = f'input/{k}'
            g_input.create_dataset(k, data=combined_data[full_key])
        g_input.attrs['key_order'] = input_order
        
        # 3. Guardar Outputs
        for k in output_order:
            full_key = f'output/{k}'
            g_output.create_dataset(k, data=combined_data[full_key])
        g_output.attrs['key_order'] = output_order
        
        # 4. Guardar Costo en la raíz (para compatibilidad con tu H5Dataset)
        f.create_dataset("C", data=combined_data['C'])

    print(f"Dataset generado exitosamente en: {output_path} (Tamaño {final_len})")