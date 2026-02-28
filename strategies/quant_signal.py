import numpy as np
import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def _circuit(x, weights):
    """
    x: np.array shape (2,)
    weights: shape (4,)
    """
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)

    # Layer 1
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)

    # Layer 2
    qml.CNOT(wires=[1, 0])
    qml.RY(weights[2], wires=0)
    qml.RY(weights[3], wires=1)

    return qml.expval(qml.PauliZ(0))

def quant_score(features, weights=None):
    """
    features: [ret_1d, norm_price_vs_sma]
    returns score in [0,1]
    """
    x = np.array(features, dtype=float)
    if x.shape != (2,):
        raise ValueError(f"quant_score expects 2 features, got shape {x.shape}")

    if weights is None:
        # Copy of the trained weights from qenv
        weights = np.array([0.56686133, -0.44826107, 0.50572016, -0.05], dtype=float)

    raw = _circuit(x, weights)
    return float(0.5 * (raw + 1.0))
