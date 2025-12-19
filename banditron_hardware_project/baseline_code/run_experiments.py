import scipy.io as sio
import numpy as np

# NO LONGER NEEDED
target_file = 'monkey_1_set_1_expt1.mat' 

try:
    print(f"--- Manual Test on {target_file} ---")
    data = sio.loadmat(target_file, simplify_cells=True)
    
    # Manually grab the keys we know exist for monkey files
    if 'feature_mat' in data:
        X = data['feature_mat']
        # We know some monkey files are missing labels, 
        # so let's check for 'label_mat' or 'target'
        Y = data.get('label_mat', data.get('target', None))
        
        if Y is None:
            print("This file has no labels! Trying an 'expt' file instead...")
            # If monkey file fails, let's try the voltage-based file
            target_file = 'expt1.mat'
            data = sio.loadmat(target_file, simplify_cells=True)
            X = data['Spike_data']
            vx = data['X_Voltage']
            vy = data['Y_Voltage']
            # Manual Quadrant Mapping
            Y = np.zeros(len(vx))
            Y[(vx >= 0) & (vy >= 0)] = 0
            Y[(vx >= 0) & (vy < 0)] = 1
            Y[(vx < 0) & (vy < 0)] = 2
            Y[(vx < 0) & (vy >= 0)] = 3

    # Ensure X is (Features, Samples)
    if X.shape[0] > X.shape[1]:
        X = X.T
        
    print(f"Data Loaded: X shape {X.shape}, Y length {len(Y)}")
    
    # Run a tiny Banditron Test
    from banditron_baseline import Banditron
    model = Banditron(n_features=X.shape[0], n_classes=int(max(Y)+1))
    results = model.train_online(X[:, :100], Y[:100])
    print(f"Banditron Test Success! Final AER: {results['final_aer']:.4f}")

except Exception as e:
    print(f"Manual test failed: {e}")
    print("Check if the file is in this exact folder!")