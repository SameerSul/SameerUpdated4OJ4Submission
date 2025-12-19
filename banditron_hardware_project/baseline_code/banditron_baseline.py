import numpy as np
import scipy.io as sio
import h5py
# baseline testing code from paper
class Banditron:
    def __init__(self, n_features, n_classes, gamma=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.gamma = gamma
        self.w = np.zeros((n_features, n_classes), dtype=np.float64)
        self.iter = 0
        self.aer_history = []
        
    def train_online(self, X, Y):
        n_samples = X.shape[1]
        error_count = 0
        for i in range(n_samples):
            self.iter += 1
            Xi = X[:, i].astype(np.float64)
            Yi = int(Y[i])
            val_f = self.w.T @ Xi
            y_hat = np.argmax(val_f)
            prob = np.full(self.n_classes, self.gamma / self.n_classes)
            prob[y_hat] += (1 - self.gamma)
            y_tilde = np.random.choice(self.n_classes, p=prob)
            if y_tilde != Yi: error_count += 1
            self.aer_history.append(error_count / self.iter)
            self.w[:, y_hat] -= Xi
            if y_tilde == Yi:
                self.w[:, Yi] += Xi / prob[y_tilde]
        return {'final_aer': self.aer_history[-1]}

    def test(self, X, Y):
        if X is None or Y is None or X.size == 0: return 0.0
        return np.mean(np.argmax(self.w.T @ X, axis=0) == Y)

def clean_numeric(data):
    """Safely converts to numpy array and ensures it is at least 1D."""
    try:
        arr = np.array(data)
        if arr.dtype.kind in 'U S O': # Skip strings/objects
            return np.array([])
        arr = arr.astype(np.float64)
        return np.atleast_1d(arr)
    except:
        return np.array([])

def robust_load(mat_file):
    try:
        data = sio.loadmat(mat_file, simplify_cells=True)
    except:
        with h5py.File(mat_file, 'r') as f:
            data = {k: np.array(f[k]) for k in f.keys() if k not in ['#refs#', '#subsystem#']}

    X, Y = None, None

    # Step 1: Search for Spike/Feature Data
    primary_keys = ['feature_mat', 'Spike_data', 'spike_data', 'features', 'spike_counts']
    for pk in primary_keys:
        if pk in data:
            X = clean_numeric(data[pk])
            if X.ndim >= 1 and X.size > 1: break
    
    # Step 2: Search nested structures (IME)
    if (X is None or X.size <= 1) and 'IMETrainingData' in data:
        s = data['IMETrainingData']
        if isinstance(s, dict):
            for k in ['features', 'spike_counts', 'Spike_data', 'spike_data']:
                X = clean_numeric(s.get(k))
                if X.size > 1:
                    Y = clean_numeric(s.get('target_direction', s.get('labels')))
                    break

    # Step 3: Handle Labels / Voltage
    if X is not None and X.size > 1:
        X = np.atleast_2d(X)
        # Ensure (Features x Samples)
        if X.shape[0] > X.shape[1] and X.shape[1] < 1000:
            X = X.T

        if Y is None or Y.size <= 1:
            # Check direct label keys
            for lk in ['labels', 'label_mat', 'target_direction', 'Y', 'target']:
                Y = clean_numeric(data.get(lk))
                if Y.size > 1: break
            
            # Check for Voltages if still no labels
            if Y is None or Y.size <= 1:
                vx = clean_numeric(data.get('X_Voltage')).flatten()
                vy = clean_numeric(data.get('Y_Voltage')).flatten()
                if vx.size > 1 and vy.size > 1:
                    min_l = min(X.shape[1], vx.size, vy.size)
                    X, vx, vy = X[:, :min_l], vx[:min_l], vy[:min_l]
                    Y = np.zeros(min_l, dtype=int)
                    Y[(vx >= 0) & (vy >= 0)] = 0
                    Y[(vx >= 0) & (vy < 0)] = 1
                    Y[(vx < 0) & (vy < 0)] = 2
                    Y[(vx < 0) & (vy >= 0)] = 3

    if X is None or X.size <= 1 or Y is None or Y.size <= 1:
        raise ValueError("Incomplete data: Neural features or labels missing/invalid.")

    # Align lengths and check classes
    min_len = min(X.shape[1], Y.size)
    X, Y = X[:, :min_len], Y[:min_len].astype(int)
    if Y.min() > 0: Y -= Y.min()
    
    if len(np.unique(Y)) < 2:
        return None, None, None, None

    split = int(X.shape[1] * 0.8)
    return X[:, :split], Y[:split], X[:, split:], Y[split:]

def run_single_experiment(dataset_path, gamma=0.01):
    try:
        X_tr, Y_tr, X_te, Y_te = robust_load(dataset_path)
        if X_tr is None:
            return {'status': 'SKIP', 'reason': 'Single class only'}
        
        model = Banditron(X_tr.shape[0], int(Y_tr.max() + 1), gamma)
        res = model.train_online(X_tr, Y_tr)
        return {
            'status': 'SUCCESS', 'train_aer': res['final_aer'],
            'test_acc': model.test(X_te, Y_te), 'dataset': dataset_path.name
        }
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}