from pprint import pprint

from src.circuits.circuit_tracer import CircuitTracer


from config import MODEL_CONFIGS, DEFAULT_CHECKPOINT_DIR

from utils.model_utils import load_model
from utils.activation_dataset_utils import load_model_dataset

model_name = 'strong'
tolerance = 1e-5


if __name__ == "__main__":
    model_file = MODEL_CONFIGS[model_name]['file']
    activation_dataset = load_model_dataset(DEFAULT_CHECKPOINT_DIR, model_name)

    model = load_model(model_file, DEFAULT_CHECKPOINT_DIR)
    tracer = CircuitTracer(model)

    # For a specific neuron of interest
    layer_name = "actor_mlp_5"
    neuron_idx = 1

    circuit = tracer.trace_circuit(layer_name, neuron_idx, activation_dataset)
