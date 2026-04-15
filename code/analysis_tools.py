import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ssm
import superlet as slt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Human-readable names for the 4 HMM behavioural states (index → label)
STATE_NAMES = {0: 'Distracted', 1: 'Deliberate', 2: 'Efficient', 3: 'Impulsive'}

def set_plot_style():
    """Apply consistent figure style for all plots."""
    sns.set_style("ticks", {
        "axes.spines.top": False,
        "axes.spines.right": False
    })
    plt.rcParams.update({
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
    })


def insert_padding(data, padding_value=-5, padding_size=50):
    # Create a DataFrame to hold the padded data
    padded_data = []
    # Get unique sessions
    unique_sessions = data['Session'].unique()

    for session in unique_sessions:
        # Extract the session data
        session_data = data[data['Session'] == session]

        # Append the session data to the padded data list
        padded_data.append(session_data)

        # Create padding DataFrame with the same session value
        padding = pd.DataFrame(padding_value, index=np.arange(padding_size), columns=data.columns)

        # Append padding to the padded data list
        padded_data.append(padding)

    # Concatenate all data
    padded_data = pd.concat(padded_data).reset_index(drop=True)

    return padded_data

def train_hmm (data, n_states, rand_seed=25, kappa=50):
    if rand_seed is not None:
        np.random.seed(rand_seed)
    np.random.seed(rand_seed)
    #stationary
    model = ssm.HMM(n_states,data.shape[1], observations="gaussian", transitions='stationary')#'sticky',transition_kwargs=dict(kappa=kappa))#
    #sticky
    #model = ssm.HMM(n_states,data.shape[1], observations="gaussian", transitions='sticky',transition_kwargs=dict(kappa=kappa))#
        
    model.fit(data,num_iters=40, init_method="kmeans")
    states = model.most_likely_states(data)
    state_means=model.observations.mus

    drop_state=np.argmin(np.mean(state_means,axis=1))#abs(-1-np.mean(state_means,axis=1)))
    drop_id= states != drop_state
    states = states[drop_id]
    state_means= np.delete(state_means, drop_state, axis=0)
    states [states == np.min(states)]=0
    states [states == np.max(states)]=1
    
    return states, state_means

def train_hmm_twice (data1,data2, rand_seed=25, kappa=50):
    #train two HMM models with 2 states
    states_1, state_means_1 = train_hmm(data1,3, rand_seed=rand_seed, kappa=kappa)
    states_2, state_means_2 = train_hmm(data2,3, rand_seed=rand_seed, kappa=kappa)
    
    #find intersections between the states to produce composite states (4)
    composite_states = states_1 + 2 * states_2
    
    means = []

    # Create combinations
    for val2 in state_means_2:
        for val1 in state_means_1:
            means.append(np.hstack((val1, val2)))

    # Convert the list of combinations to a NumPy array
    composite_means = np.array(means)
    
    state_means, states,states_accuracy,states_speed, states_map = reassign_states(composite_means, composite_states, data1.shape[1], data2.shape[1])   
    np.random.seed(None)
    return states, state_means,states_accuracy,states_speed

def prepare_and_train_dataset(dummy_index, data, accuracy_metrics, speed_metrics, kappa=50, shuffle=True, window=5):
    training_data = []
    for sesh_dict in data:
        sesh = pd.DataFrame.from_dict(sesh_dict)  # Convert dictionary to dataframe
        sesh = sesh.copy()
        if shuffle:
            sesh = sesh.sample(frac=1).reset_index(drop=True)

        sesh = detrend_learning(sesh, ['TD','PL','Prec', 'RT', 'SpeedMean'])
        sesh = smoothing(session=sesh, window=window, metrics=['Correct', 'Wrong', 'Bias', 'Prec','TD','PL', 'RT', 'SpeedMean'])
        sesh = scaling(session=sesh, scaler='standard', metrics=['~Correct_Raw','Wrong_Raw', 'Session', 'Bias_Raw', 'MorphTarget', 'MorphDistractor', 'Outcome', 'TrialDuration','TrialIndex'])
        training_data.append(sesh)  # Keep as dataframe for now
    training_data = concatenating(training_data)
    min_val = training_data[accuracy_metrics + speed_metrics].min().min()
    training_data = insert_padding(training_data, padding_value=min_val - 0.001)

    training_acc = training_data[accuracy_metrics].values
    training_sp = training_data[speed_metrics].values
    
    training_data = training_data[training_data['Session'] >= 0].reset_index(drop=True)
    # Add HMM states to the training data
    states, state_means, states_accuracy, states_speed = train_hmm_twice(training_acc, training_sp, rand_seed= 25, kappa=kappa)
    training_data['States'] = states
    training_data['States_Accuracy'] = states_accuracy
    training_data['States_Speed'] = states_speed

    return training_data#.columns.tolist(), training_data.values  # Return as a tuple of columns and values


def train_hmm_single(X, n_states=5, rand_seed=0):
    """
    Fit ONE sticky Gaussian HMM on ALL features.
    Use n_states=5 so one state can soak up the padding; we drop it after decoding.
    Returns (states_without_padding, state_means_without_padding).
    """
    np.random.seed(rand_seed)
    model = ssm.HMM(
        n_states,              # K
        X.shape[1],            # D
        observations="gaussian",
        transitions="stationary",
        # transitions="sticky",
        # transition_kwargs=dict(kappa=50),
    )
    model.fit(X, num_iters=40, init_method="kmeans")
    states = model.most_likely_states(X)
    mus = model.observations.mus

    # Drop the padding state (the globally lowest-mean state across dims)
    drop_state = np.argmin(np.mean(mus, axis=1))
    keep_mask = states != drop_state
    states = states[keep_mask]
    mus = np.delete(mus, drop_state, axis=0)

    # Relabel remaining states to 0..N-1
    uniq = np.sort(np.unique(states))
    remap = {old: i for i, old in enumerate(uniq)}
    states = np.fromiter((remap[s] for s in states), dtype=int)

    np.random.seed(None)
    return states, mus

def prepare_and_train_dataset_single(dummy_index, data, accuracy_metrics, speed_metrics,
                                     n_states=5, rand_seed=25, shuffle=False):
    """
    Preprocess sessions and fit ONE sticky Gaussian HMM with `n_states` on ALL features (accuracy + speed).
    Uses a padding state and drops it post hoc (mirrors your 2-HMM behavior).
    Returns a DataFrame with a single 'States' column.
    """
    # --- collect & preprocess per-session ---
    training_data = []
    for sesh_dict in data:
        sesh = pd.DataFrame.from_dict(sesh_dict).copy()
        if shuffle:
            sesh = sesh.sample(frac=1, random_state=rand_seed).reset_index(drop=True)

        sesh = detrend_learning(sesh, ['TD','PL','Prec', 'RT', 'SpeedMean'])
        sesh = smoothing(session=sesh, window=5, metrics=['Correct', 'Bias', 'Prec','TD','PL', 'RT', 'SpeedMean'])
        sesh = scaling(session=sesh, scaler='standard',
                       metrics=['~Correct_Raw', 'Session', 'Bias_Raw', 'MorphTarget', 'MorphDistractor',
                                'Outcome', 'TrialDuration', 'TrialIndex'])
        training_data.append(sesh)

    # --- concat + padding (same convention as your working pipeline) ---
    training_data = concatenating(training_data)
    min_val = training_data[accuracy_metrics + speed_metrics].min().min()
    training_data = insert_padding(training_data, padding_value=min_val - 0.001)

    # --- feature matrix for HMM (includes padding rows) ---
    X = np.hstack([
        training_data[accuracy_metrics].values,
        training_data[speed_metrics].values
    ])

    # --- fit ONE HMM and drop padding state ---
    states, state_means = train_hmm_single(X, n_states=n_states, rand_seed=rand_seed)

    # --- drop padding rows from DF to align lengths (Session<0 were padding) ---
    training_data = training_data[training_data['Session'] >= 0].reset_index(drop=True)

    # --- attach single-state labels ---
    training_data['States'] = states

    return training_data


def compute_superlets_per_sesh(data,sesh_id, fs, foi):
    power_per_sesh=[]
    for isesh in np.unique(sesh_id):
        sesh=data[sesh_id==isesh].copy()
        shuffled_power = slt.superlets(sesh[np.newaxis,:], fs, foi)
        power_per_sesh.append(shuffled_power)
    #power_per_sesh=np.concatenate(power_per_sesh,axis=1)
    return power_per_sesh

def reassign_states(means, states, n_accuracy_metrics, n_speed_metrics,manual_map=None):
    means=np.array(means)
    try:
        if manual_map:
            states_map = manual_map
        else:
            # Split the means into accuracy and speed
            accuracy_means = means[:, :n_accuracy_metrics]
            speed_means = means[:, n_accuracy_metrics:n_accuracy_metrics + n_speed_metrics]
            
            # Calculate the mean of accuracy and speed for each state
            accuracy_means_avg = np.mean(accuracy_means, axis=1)
            speed_means_avg = np.mean(speed_means, axis=1)

            accuracy_means_max = accuracy_means_avg>np.mean(accuracy_means_avg)#accuracy_means_avg==np.max(accuracy_means_avg)
            accuracy_means_min = accuracy_means_avg<np.mean(accuracy_means_avg)#accuracy_means_avg==np.min(accuracy_means_avg)
            speed_means_max = speed_means_avg>np.mean(speed_means_avg)#speed_means_avg==np.max(speed_means_avg)
            speed_means_min = speed_means_avg<np.mean(speed_means_avg) #speed_means_avg==np.min(speed_means_avg)
        
            states_map = {
                0:np.where((accuracy_means_min)&(speed_means_min))[0][0],
                1:np.where((accuracy_means_max)&(speed_means_min))[0][0],
                2:np.where((accuracy_means_max)&(speed_means_max))[0][0],
                3:np.where((accuracy_means_min)&(speed_means_max))[0][0]
                }

        # Create a new state means array with states ordered by quadrants
        new_state_means = np.array([means[states_map[i]] for i in range(4)])
        #print(np.array([means[state_to_quadrant[i]] for i in range(4)]))

        new_states=np.empty(len(states))
        new_states_speed=np.empty(len(states))
        new_states_accuracy=np.empty(len(states))
        for new_state in states_map:
            old_state=states_map[new_state]
            new_states[np.where(states==old_state)[0]]=new_state
            if new_state == 0:
                new_states_accuracy[np.where(states==old_state)[0]]=0
                new_states_speed[np.where(states==old_state)[0]]=0
            elif new_state == 1:
                new_states_accuracy[np.where(states==old_state)[0]]=1
                new_states_speed[np.where(states==old_state)[0]]=0
            elif new_state == 2:
                new_states_accuracy[np.where(states==old_state)[0]]=1
                new_states_speed[np.where(states==old_state)[0]]=1
            else:
                new_states_accuracy[np.where(states==old_state)[0]]=0
                new_states_speed[np.where(states==old_state)[0]]=1

    except Exception as e:
        # Handle exceptions and fallback to the original means and states
        print("An error occurred:", str(e))
        print("Returning original means and states.")
        new_state_means = means
        new_states = states
        new_states_accuracy = states
        new_states_speed = states
        states_map = {i: i for i in range(len(means))}
        
    return new_state_means, new_states,new_states_accuracy,new_states_speed, states_map

def select_metrics(session,metrics):
    if not metrics:
        metrics=session.columns
    if '~' in metrics[0]:
        metrics[0]=metrics[0][1:]
        metrics = session.columns[~session.columns.isin(metrics)].tolist()
    else:
        metrics = metrics
    return metrics

def concatenating (sessions, sesh_id=False):
    return pd.concat(sessions, ignore_index=True)

def detrend_learning (session, metrics):

    for metric in metrics:
        X = session[metric].index
        X = np.reshape(X, (len(X), 1))
        y = session[metric].values

        pf = PolynomialFeatures(degree=2)
        Xp = pf.fit_transform(X)
        md2 = LinearRegression()
        md2.fit(Xp, y)
        trendp = md2.predict(Xp)
        detrpoly = [y[i] - trendp[i] for i in range(0, len(y))]
        session[metric] = pd.array(detrpoly, dtype=float)
    return session

def smoothing (session, window, metrics=None):
    metrics=select_metrics(session,metrics)
    session=session.copy()   
    for metric in metrics:
        session[metric] = session[metric].astype(float)
        session[metric+'_Raw']=session[metric]
    session.loc[:,metrics]=session.loc[:,metrics].rolling(window=window, center=True).mean()
    session.dropna(inplace=True)    
    return session

def scaling (session, metrics, scaler='standard'):
    metrics=select_metrics(session,metrics)
    session=session.copy()
    if scaler=='robust':
        scaler=RobustScaler()
    elif scaler=='standard':
        scaler=StandardScaler()
    elif scaler=='minmax':
        scaler=MinMaxScaler()
    else:
        print('Invalid scaler name')
    for col in metrics:
        session[col] = session[col].astype(float)
    session.loc[:,metrics] = scaler.fit_transform(session.loc[:,metrics])
    return session

def read_dfs (folder, species_list=['human', 'monkey', 'mouse']):
    """
    Load lists of DataFrames for multiple species from pickle files.

    Args:
        folder (str): Base folder name for the files.
        species_list (list): List of species identifiers to load.

    Returns:
        dict: A dictionary where keys are species names and values are lists of DataFrames.
    """
    data_frames = {}

    # Assumes CWD is behaviour/code/ — data lives at ../data/
    parent_path, _ = os.path.split(os.getcwd())
    savepath_base = os.path.join(parent_path, 'data')

    for species in species_list:
        file_path = os.path.join(savepath_base, f'data_{folder}_{species}.pkl')
        # Check if the pickle file exists
        if not os.path.exists(file_path):
            print(f"Warning: Pickle file not found for species {species}. Skipping.")
            continue

        # Load the list of DataFrames for this species
        with open(file_path, 'rb') as f:
            data_frames[species] = pickle.load(f)
    
    return data_frames
    
def find_learning_cutoff(df, col='RT', window=10,threshold=0.5, consecutive=5):
    half_size = len(df) // 2
    stable_mean = df[col].iloc[-half_size:].mean()
    stable_std = df[col].iloc[-half_size:].std()
    rolling_avg = df[col].rolling(window=window).mean()

    consecutive_count = 0
    for i in range(len(rolling_avg)):
        if abs(rolling_avg.iloc[i] - stable_mean) < threshold * stable_std:
            consecutive_count += 1
        else:
            consecutive_count = 0
        if consecutive_count >= consecutive:
            return max(0, i - consecutive + 1)  # cutoff index

    return 0  # default if never meets criterion

def find_combined_learning_cutoff(df, window=10, threshold=0.5, consecutive=5):
    rt_cutoff = find_learning_cutoff(df, col='RT', window=window, threshold=threshold, consecutive=consecutive)
    speed_cutoff = find_learning_cutoff(df, col='SpeedMean', window=window, threshold=threshold, consecutive=consecutive)
    
    # Take the maximum cutoff to ensure both metrics are stable
    return max(rt_cutoff, speed_cutoff)
    
def save_dfs(data_list, filename, species, savepath=None):
    """
    Save a list of DataFrames with their metadata as a pickle file.

    Args:
        data_list (list): List of DataFrames with attributes.
        filename (str): Base name for the file.
        species (str): Species identifier.
        savepath (str): Directory to save the file.
    """
    if not savepath:
        parent_path, _ = os.path.split(os.getcwd())
        savepath = os.path.join(parent_path, 'data')
    
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    
    file_path = os.path.join(savepath, f'data_{filename}_{species}.pkl')
    
    # Save the list of DataFrames as a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data_list, f)

def save_pickle(data, filename,species,savepath=None, verbose=False):
    # Assumes CWD is behaviour/code/ — data lives at ../data/
    if not savepath:
        parent_path, _ = os.path.split(os.getcwd())
        savepath=os.path.join(parent_path,'data')
    if not os.path.isdir(savepath):
        os.makedirs(savepath)   
        
    file_path = os.path.join(savepath, f'data_{filename}_{species}.pkl') 
    
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    if verbose:
        print(f"Data saved to {file_path}")
       
def load_pickle(filename,species,savepath=None, verbose=False):
    # Assumes CWD is behaviour/code/ — data lives at ../data/
    if not savepath:
        parent_path, _ = os.path.split(os.getcwd())
        savepath=os.path.join(parent_path,'data')
            
    file_path = os.path.join(savepath, f'data_{filename}_{species}.pkl') 

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    if verbose:
        print(f"Data loaded from {file_path}")
    return data

def save_plot(filename="plot.png"):
    """Save the current plot to behaviour_paper/figures/."""
    parent_path, _ = os.path.split(os.getcwd())
    savepath = os.path.join(parent_path, 'figures')
    os.makedirs(savepath, exist_ok=True)
    plt.savefig(os.path.join(savepath, filename), format='png', dpi=300)
        
def pretty_label(col):
    # strip common suffixes
    base = (
        col.replace("_z_smooth","")
           .replace("_z","")
           .replace("_raw","")
           .replace("_Raw","")
           .replace("_clean","")
    )

    mapping = {
        "Correct":   "Hit rate",
        "Prec":      "Precision",
        "Bias":      "Bias (inv.)",
        "SpeedMean": "Speed",
        "RT":        "RT (inv.)",
        "TD":        "Target distance",
        "PL":        "Path length",
        "MorphFold": "Difficulty",
        "mean_0_200": "Pupil diameter",
        "mean_prev_m200_0": "Pupil diameter",
        "mean_3042_3043": "Pupil diameter",
    }

    return mapping.get(base, base)# base

def loader(sp):
    # Must return a DataFrame with columns: 'States' (int) and optional 'Session' (hashable)
    return load_pickle('hmm_sessions', sp)

def rle_with_delays(inarray):
    """ Run length encoding with delay times between runs of the same value.
        returns: tuple (runlengths, startpositions, delaytimes, values) """
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None, None)
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element position
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        values = ia[i]
        
        # Initialize delay times array with zeros
        d = np.zeros(len(z), dtype=int)
        # Track the last end position for each value
        last_end_positions = {}

        for idx, val in enumerate(values):
            if val in last_end_positions:
                # Calculate delay as current start position - last end position of the same value
                delay = p[idx] - last_end_positions[val]
                d[idx] = delay
            # Update last end position for the value
            last_end_positions[val] = p[idx] + z[idx]
        
        # For the first occurrence of each value, delay is set to 0 by default
        return (z, p, d, values)

def rle(a):
    """
    Basic run-length encoding.
    Returns (z, p, values):
      z: run lengths (counts)
      p: start indices (0-based)
      values: label at each run
    """
    a = np.asarray(a)
    if a.size == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=a.dtype))
    change = np.r_[True, a[1:] != a[:-1]]
    p = np.where(change)[0]              # run starts
    z = np.diff(np.r_[p, a.size])        # run lengths
    values = a[p]                        # labels
    return z, p, values


def rle_plus(a, durations=None, time_seconds=False):
    z, p, values = rle(a)
    n_runs = len(z)
    if n_runs == 0:
        return z, p, np.array([], dtype=float if time_seconds else int), values

    if time_seconds:
        # seconds: float dtype
        t = np.r_[0.0, np.asarray(durations, float).cumsum()]
        start_t = t[p]
        end_t   = t[p + z]
        d = np.zeros(n_runs, dtype=float)                      # first delay = 0.0
        last_end = {}
        for k, lab in enumerate(values):
            d[k] = start_t[k] - last_end[lab] if lab in last_end else 0.0
            last_end[lab] = end_t[k]
        return z, p, d, values
    else:
        # counts: int dtype
        start_i = p
        end_i   = p + z
        d = np.zeros(n_runs, dtype=int)                        # first delay = 0
        last_end = {}
        for k, lab in enumerate(values):
            d[k] = start_i[k] - last_end[lab] if lab in last_end else 0
            last_end[lab] = end_i[k]
        return z, p, d, values

def shuffle_blocks(states, rng=None):
    """Permute RLE blocks; preserves run lengths and labels, shuffles order."""
    s = np.asarray(states)
    if rng is None: rng = np.random.default_rng()
    L, _, V = rle(s)
    if L.size == 0:
        return s.copy()
    perm = rng.permutation(len(L))
    out = np.empty_like(s)
    pos = 0
    for length, val in zip(L[perm], V[perm]):
        out[pos:pos+length] = val
        pos += length
    return out

def transition_counts(states, num_states=None, window=1):
    """
    SxS transition COUNT matrix.
    For each t, add counts from s[t] to s[t+1],...,s[t+window] (if within bounds).
    """
    s = np.asarray(states, int)
    S = int(np.max(s)) + 1 if num_states is None else int(num_states)
    C = np.zeros((S, S), float)
    n = len(s)
    for t in range(n):
        src = s[t]
        end = min(n, t + window + 1)
        for nxt in s[t+1:end]:
            C[src, int(nxt)] += 1.0
    return C

def normalize_rows(C, minus_diag=True, eps=1e-12):
    """Row-normalize counts to probabilities. Optionally zero (and ignore) the diagonal."""
    M = C.astype(float).copy()
    if minus_diag:
        np.fill_diagonal(M, 0.0)
    rowsum = M.sum(axis=1, keepdims=True)
    M = np.divide(M, np.maximum(rowsum, eps), where=np.ones_like(M, dtype=bool))
    return M

def transition_prob_matrix(states, num_states=None, window=1, minus_diag=True):
    return normalize_rows(transition_counts(states, num_states, window), minus_diag=minus_diag)

    
def create_transition_matrix(states, minus_diag=False, window=1):

    num_states = len(np.unique(states))
    transition_matrix = np.zeros((num_states, num_states))
    for itr,tr in enumerate(states):
        for jtr in range(1,window+1):
            if itr + jtr < len(states):
                transition_matrix[int(tr), int(states[itr+jtr])] += 1
    # Normalize the rows to sum to 1
    transition_matrix = np.nan_to_num(transition_matrix / transition_matrix.sum(axis=1, keepdims=True))
    
    if minus_diag:
        # Subtract the diagonal
        diag_matrix = np.diag(np.diag(transition_matrix))
        modified_matrix = transition_matrix - diag_matrix

        # Normalize off-diagonal terms
        row_sums = modified_matrix.sum(axis=1) - np.diag(modified_matrix)
        transition_matrix = np.divide(modified_matrix, row_sums[:, np.newaxis], out=np.zeros_like(modified_matrix), where=row_sums[:, np.newaxis] != 0)

    
    return transition_matrix


def plot_states_timeseries(session_data, colormap_states, species,
                           state_col='States', max_trials=None):
    """Plots the states sequence for each session, with optional trial cutoff."""
    session_id = session_data['Session']
    unique_sessions = np.unique(session_id)
    num_sessions = len(unique_sessions)

    fig, axs = plt.subplots(num_sessions, 1, figsize=(15, num_sessions), sharex=True)
    if num_sessions == 1:
        axs = np.array([axs])

    fig.suptitle(f'States timeline, {species}')
    sesh_lens = []

    for isesh in unique_sessions:
        ax = axs[int(isesh)]
        session_states = session_data[state_col][session_id == isesh].reset_index(drop=True).values

        if max_trials is not None:
            session_states = session_states[:max_trials]

        sesh_lens.append(len(session_states))

        unique_states = np.unique(session_states)
        num_states = len(unique_states)

        ax.imshow(session_states[np.newaxis, :], cmap=colormap_states, aspect='auto')
        ax.set_yticks([])
        ax.set_ylabel(f'sesh {isesh}')

        handles = [
            plt.Rectangle((0, 0), 1, 1,
                          color=colormap_states(i / num_states),
                          label=STATE_NAMES.get(i, f'State {i}'))
            for i in unique_states
        ]

    if max_trials is not None:
        for ax in axs:
            ax.set_xlim(0, max_trials)

    fig.legend(handles, [h.get_label() for h in handles],
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()




def aggregate_species_counts(df, num_states, window=1, session_weighting=False):
    """Returns a list of per-session transition COUNT matrices."""
    if 'Session' in df.columns:
        sessions = [g['States'].values for _, g in df.groupby('Session')]
    else:
        sessions = [df['States'].values]
    mats = [transition_counts(s, num_states=num_states, window=window) for s in sessions]
    if session_weighting:
        mats = [normalize_rows(m, minus_diag=True) for m in mats]
    return mats


def cohen_h(p1, p2):
    """Cohen's h effect size for two proportions."""
    p1 = np.clip(p1, 0, 1)
    p2 = np.clip(p2, 0, 1)
    return 2.0 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))



def mouse_key_from_session(session):
    """Extract a unique key (Animal-Date) from a session's metadata attributes."""
    md = getattr(session, "attrs", {}).get("Metadata", {})
    date = str(md.get("Start Time", ""))[:10].replace("-", "")
    subj = md.get("Subject", md.get("Animal", "")).strip()
    if subj and not subj.startswith("Animal"):
        subj = f"Animal{subj}"
    if not (date and subj):
        raise ValueError(f"Missing date/subject in session metadata: {md}")
    return f"{subj}-{date}"


def _events_in_mouse_trial(evt_ts, evt_codes, code_mask, t0):
    """Return list of (t_rel, code) tuples for events matching code_mask within a trial."""
    if not np.any(code_mask):
        return []
    ts = (evt_ts[code_mask] - t0).astype(float)
    cs = evt_codes[code_mask].astype(int)
    return list(zip(ts, cs))


def annotate_reward_history(trials):
    """
    Add cross-trial reward history fields to a list of trial dicts.
    Each trial must have 't_abs0' and 'events' with optional 'reward_5xxx' key.
    """
    last_reward_end_abs = np.nan
    last_reward_duration = np.nan
    last_reward_trial_idx = None
    for k, tr in enumerate(trials):
        t0 = tr["t_abs0"]
        gap = (t0 - last_reward_end_abs) if np.isfinite(last_reward_end_abs) else np.nan
        n_since = (k - last_reward_trial_idx) if last_reward_trial_idx is not None else np.nan
        tr["time_since_last_reward_s"] = float(gap) if np.isfinite(gap) else np.nan
        tr["trials_since_last_reward"] = (int(n_since) if np.isfinite(n_since) else np.nan)
        tr["last_reward_duration_s"]   = float(last_reward_duration) if np.isfinite(last_reward_duration) else np.nan
        tr["prev_rewarded_immediate"]  = (
            1.0 if np.isfinite(n_since) and int(n_since) == 1
            else (0.0 if np.isfinite(n_since) else np.nan)
        )
        rewards = tr["events"].get("reward_5xxx", [])
        if rewards:
            start_ts, code = rewards[-1]
            dur_s = (code - 5000) / 1000.0
            last_reward_end_abs = t0 + float(start_ts) + dur_s
            last_reward_duration = dur_s
            last_reward_trial_idx = k


def _window_mean_allowing_nans(t, p, a, b, max_nan_s, enforce_nan_limit=True):
    """Median pupil in [a, b], with optional NaN-fraction quality gate."""
    if t.size == 0 or a >= b:
        return np.nan
    m = (t >= a) & (t <= b)
    if not m.any():
        return np.nan
    vals = np.asarray(p[m], float)
    if enforce_nan_limit and max_nan_s is not None:
        dt = np.nanmedian(np.diff(t)) if t.size > 1 else 0.002
        max_nan_n = int(max_nan_s / max(dt, 1e-9))
        if np.isnan(vals).sum() > max_nan_n:
            return np.nan
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.nanmedian(vals))


def extract_pupil_features_from_trials(trials, session_index, session_tag,
                                       win=0.2, max_nan_s=0.05, enforce_nan_limit=True):
    """
    Generic trial → pupil feature extractor (shared for monkey & mouse).
    trials: list of dicts with 't_abs0', 'time', 'pupil', 'events', optionally 'TrialIndex'.
    Returns list of row dicts with pupil window means and reward history.
    """
    rows = []
    annotate_reward_history(trials)
    for k, tr in enumerate(trials):
        t = np.asarray(tr["time"], float)
        p = np.asarray(tr["pupil"], float)
        mean_0_200 = _window_mean_allowing_nans(t, p, 0.0, win, max_nan_s, enforce_nan_limit=enforce_nan_limit)
        if not np.isfinite(mean_0_200):
            continue
        mean_prev = np.nan
        if k > 0 and trials[k-1]["time"].size:
            tprev = np.asarray(trials[k-1]["time"], float)
            pprev = np.asarray(trials[k-1]["pupil"], float)
            mean_prev = _window_mean_allowing_nans(
                tprev, pprev, tprev[-1] - win, tprev[-1], max_nan_s, enforce_nan_limit=enforce_nan_limit)
        ev = tr["events"]
        t42 = ev.get("stim_start_3042", [])
        t43 = ev.get("stim_stop_3043", [])
        if t42 and t43:
            a, b = t42[0][0], t43[0][0]
            mean_3042_3043 = (
                _window_mean_allowing_nans(t, p, a, b, max_nan_s, enforce_nan_limit=enforce_nan_limit)
                if b > a else np.nan
            )
        else:
            mean_3042_3043 = np.nan
        rows.append({
            "Session": session_index, "session_tag": session_tag,
            "TrialIndex": tr.get("TrialIndex", k),
            "mean_0_200": mean_0_200, "mean_prev_m200_0": mean_prev, "mean_3042_3043": mean_3042_3043,
            "time_since_last_reward_s": tr.get("time_since_last_reward_s", np.nan),
            "trials_since_last_reward": tr.get("trials_since_last_reward", np.nan),
            "last_reward_duration_s":   tr.get("last_reward_duration_s", np.nan),
            "prev_rewarded_immediate":  tr.get("prev_rewarded_immediate", np.nan),
        })
    return rows


def build_mouse_eye_trial_features(species_list, mouse_pupil, out_pkl=None,
                                   win=0.2, max_nan_s=0.05):
    """
    Build per-trial pupil feature rows for mouse sessions matched to eye tracking data.
    Sessions without a matching key in mouse_pupil are silently skipped.
    """
    rows = []
    for i_ses, ses in enumerate(species_list):
        try:
            key = mouse_key_from_session(ses)
        except ValueError:
            continue
        if key not in mouse_pupil:
            continue
        ses_sorted = ses.sort_values("TrialIndex").reset_index(drop=True) if "TrialIndex" in ses.columns else ses.reset_index(drop=True)
        pupil_trials = mouse_pupil[key]["PupilDia"]
        n = min(len(ses_sorted), len(pupil_trials))
        if n == 0:
            continue
        trials = []
        for j in range(n):
            row = ses_sorted.iloc[j]
            evt_codes = np.asarray(row["Event"], int)
            evt_ts    = np.asarray(row["EventTs"], float)
            uniq_ts = np.unique(evt_ts)
            if uniq_ts.size == 0:
                continue
            pup = np.asarray(pupil_trials[j], float)
            if len(pup) != len(uniq_ts):
                m = min(len(pup), len(uniq_ts))
                pup = pup[:m]; uniq_ts = uniq_ts[:m]
            t0 = uniq_ts[0]
            t_rel = uniq_ts - t0
            ev = {
                "stim_start_3042": _events_in_mouse_trial(evt_ts, evt_codes, evt_codes == 3042, t0),
                "stim_stop_3043":  _events_in_mouse_trial(evt_ts, evt_codes, evt_codes == 3043, t0),
                "reward_5xxx":     _events_in_mouse_trial(evt_ts, evt_codes, (evt_codes >= 5000) & (evt_codes < 6000), t0),
                "response_1_2":    _events_in_mouse_trial(evt_ts, evt_codes, (evt_codes == 1) | (evt_codes == 2), t0),
            }
            trials.append({
                "t_abs0": float(t0), "time": t_rel.astype(float),
                "pupil": pup.astype(float), "events": ev, "TrialIndex": j,
            })
        rows.extend(extract_pupil_features_from_trials(trials, i_ses, key, win=win, max_nan_s=max_nan_s,
                                                       enforce_nan_limit=False))
    df = pd.DataFrame(rows)
    if out_pkl is not None:
        with open(out_pkl, "wb") as f:
            pickle.dump(rows, f)
    return df


def build_monkey_eye_trial_features(in_pkl, out_pkl, win=0.2, max_nan_s=0.05):
    """
    Extract per-trial pupil features for monkey sessions.
    in_pkl:  path to intermediate eye_trials.pkl (produced by build_trials_pkl)
    out_pkl: path to write monkey_eye_trials_features.pkl
    """
    with open(in_pkl, "rb") as f:
        sessions = pickle.load(f)
    rows = []
    for i_ses, ses in enumerate(sessions):
        tag = ses["session_tag"]
        T   = ses["trials"]
        for k, tr in enumerate(T):
            tr.setdefault("TrialIndex", k)
        rows.extend(extract_pupil_features_from_trials(
            T, session_index=i_ses, session_tag=tag,
            win=win, max_nan_s=max_nan_s, enforce_nan_limit=True,
        ))
    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(rows, f, protocol=4)
    print(f"Saved {len(rows)} monkey rows → {out_pkl}")


def _z_per_session(df, cols, session_col="Session", suffix="_z"):
    """Z-score specified columns within each session."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            z = df.groupby(session_col)[c].transform(lambda v: (v - v.mean()) / v.std(ddof=1))
            df[f"{c}{suffix}"] = z.replace([np.inf, -np.inf], np.nan)
    return df


def _concat_species(species_list_dfs):
    """Concatenate a list of session DataFrames, adding a Session index column."""
    frames = []
    for i, d in enumerate(species_list_dfs):
        dd = d.copy()
        if "Session" not in dd:
            dd["Session"] = i
        frames.append(dd)
    return pd.concat(frames, ignore_index=True)


def _load_eye_features(pkl):
    """Load eye feature rows from a pickle file and return as DataFrame."""
    with open(pkl, "rb") as f:
        return pd.DataFrame(pickle.load(f))


def build_unified_df(*, species, species_data_dict, eye_features_pkl, boundary=50):
    """
    Combine raw species data, z-scored training data, and pupil eye features
    into one unified DataFrame for pupil–behaviour analysis.

    Parameters
    ----------
    species : str  (e.g. 'monkey' or 'mouse')
    species_data_dict : dict  (the species_data dict loaded from sat.read_dfs)
    eye_features_pkl : str or Path  (path to pickled eye feature rows)
    boundary : int  (morph value for folding; default 50)
    """
    species_list = species_data_dict[species]
    S = _concat_species(species_list)
    keys = ["Session", "TrialIndex", "MorphTarget"]
    metrics = ["Correct", "RT", "SpeedMean", "PL", "TD", "Prec", "Bias"]
    keep_cols = list(dict.fromkeys(keys + metrics + [c for c in S.columns if c.startswith("Session_Info")]))
    S = S[[c for c in keep_cols if c in S.columns]].copy()
    S = S.rename(columns={m: f"{m}_raw" for m in metrics if m in S.columns})

    T = load_pickle("hmm_sessions", species=species).copy()
    rename_z, rename_z_smooth = {}, {}
    for m in metrics:
        if f"{m}_Raw" in T.columns: rename_z[f"{m}_Raw"] = f"{m}_z"
        if m in T.columns:          rename_z_smooth[m]   = f"{m}_z_smooth"
    T = T.rename(columns={**rename_z, **rename_z_smooth})
    state_cols = [c for c in T.columns if c.startswith("States")]
    T = T[[c for c in (keys + list(rename_z.values()) + list(rename_z_smooth.values()) + state_cols) if c in T.columns]].copy()

    df = S.merge(T, on=["Session", "TrialIndex", "MorphTarget"], how="inner", validate="one_to_one")

    E = _load_eye_features(eye_features_pkl)
    df = df.merge(E, on=["Session", "TrialIndex"], how="inner", validate="one_to_one")

    pupil_cols = [c for c in ["mean_0_200", "mean_prev_m200_0", "mean_3042_3043"] if c in df.columns]
    if pupil_cols:
        df = _z_per_session(df, pupil_cols, session_col="Session", suffix="_z")

    df["MorphFold"] = np.abs(df["MorphTarget"] - boundary)

    if "Wrong_raw" in df.columns:
        df["Wrong_prev"] = df.groupby("Session")["Wrong_raw"].shift(1)
    elif "Correct_raw" in df.columns:
        df["Wrong_prev"] = 1.0 - df.groupby("Session")["Correct_raw"].shift(1)

    sp_col = next((c for c in ["SpeedMean_z", "SpeedMean_z_smooth", "SpeedMean_raw"] if c in df.columns), None)
    if sp_col is not None:
        df["SpeedMean_prev"] = df.groupby("Session")[sp_col].shift(1)

    core = ["Session", "TrialIndex", "MorphTarget"]
    df = df.dropna(subset=[c for c in core if c in df.columns]).reset_index(drop=True)
    return df


def gsem(x):
    """Grand SEM: std / sqrt(n) across non-NaN values."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def residualize_pupil(df, pupil_cols, predictors=(
        "time_since_last_reward_s", "trials_since_last_reward",
        "last_reward_duration_s", "prev_rewarded_immediate", "MorphFold"),
        by="Session"):
    """Regress out predictors from pupil columns (within sessions) and return residuals."""
    D = df.copy()
    preds = [p for p in predictors if p in D.columns]
    if not preds:
        return D
    for ses, G in D.groupby(by):
        X = G[preds].astype(float).dropna()
        if X.empty:
            continue
        X_ = np.c_[np.ones(len(X)), X.values]
        for pc in pupil_cols:
            if pc not in G:
                continue
            y = G.loc[X.index, pc].astype(float)
            m = y.notna()
            if m.sum() < 20:
                continue
            beta, *_ = np.linalg.lstsq(X_[m], y[m].values, rcond=None)
            yhat = X_.dot(beta)
            resid = pd.Series(np.nan, index=G.index)
            resid.loc[X.index] = G.loc[X.index, pc] - yhat
            D.loc[G.index, f"{pc}_clean"] = resid
    return D


def _bin_stats(df, bin_col, val_col, *, q=12, min_per_bin=30,
               max_levels=2, equal_positions=False):
    """Bin val_col by bin_col quantiles and return (centers, means, SEMs, labels)."""
    D = df[[bin_col, val_col]].dropna().copy()
    if D.empty:
        return None
    uniq = np.sort(pd.unique(D[bin_col]))
    if len(uniq) <= max_levels:
        G = D.groupby(bin_col); N = G.size()
        keep = [u for u in uniq if u in N.index and N.loc[u] >= min_per_bin]
        if not keep:
            return None
        centers = np.array(keep, dtype=float)
        y = np.array([G[val_col].mean().loc[v] for v in keep])
        e = np.array([gsem(G[val_col].get_group(v)) for v in keep])
        if equal_positions:
            x = np.arange(len(keep), dtype=float)
            labels = [str(v) for v in centers]
            return x, y, e, labels
        return centers, y, e, None
    for qb in range(q, 1, -1):
        try:
            D["_b"] = pd.qcut(D[bin_col], q=qb, labels=False, duplicates="drop")
            break
        except ValueError:
            continue
    if "_b" not in D:
        return None
    G = D.groupby("_b")
    N = G.size(); keep = N[N >= min_per_bin].index
    if len(keep) == 0:
        return None
    centers = G[bin_col].median().loc[keep].values
    y = G[val_col].mean().loc[keep].values
    e = G[val_col].apply(gsem).loc[keep].values
    if equal_positions:
        x = np.arange(len(centers), dtype=float)
        labels = [f"{v:.3g}" for v in centers]
        return x, y, e, labels
    return centers, y, e, None


def _smooth_bin_stats(df, bin_col, val_col, *, n_centers=20, width_frac=0.3,
                      min_per_bin=30, agg="mean", error="sem", ci_q=(16, 84)):
    """Overlapping sliding-window binning of val_col by bin_col."""
    D = df[[bin_col, val_col]].dropna().copy()
    if D.empty:
        return None
    x = D[bin_col].values.astype(float)
    y = D[val_col].values.astype(float)
    if len(x) == 0:
        return None
    x_min, x_max = np.min(x), np.max(x)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return None
    centers = np.linspace(x_min, x_max, n_centers)
    width   = width_frac * (x_max - x_min)
    xs, ys, lo, hi = [], [], [], []
    for c in centers:
        m = (x >= c - width/2) & (x <= c + width/2)
        if m.sum() < min_per_bin:
            continue
        vals = y[m]
        stat = np.mean(vals) if agg == "mean" else np.median(vals)
        if error == "ci":
            q_lo, q_hi = np.percentile(vals, [ci_q[0], ci_q[1]])
            lo.append(q_lo); hi.append(q_hi)
        else:
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
            lo.append(stat - sem); hi.append(stat + sem)
        xs.append(c); ys.append(stat)
    if not xs:
        return None
    return np.asarray(xs), np.asarray(ys), np.asarray(lo), np.asarray(hi)


def plot_pupil_suite_minimal(
    df, species='unknown', *,
    pupil_cols=("mean_0_200_z",),
    groups=None, q=12, min_per_bin=30,
    equal_spacing_top=True,
    regress_reward=False,
    predictors=("time_since_last_reward_s", "last_reward_duration_s"),
    smooth=False, smooth_centers=25, smooth_width_frac=0.3,
    agg="median", ci_q=(16, 84),
    bottom_pupil_axis="x",
    mode="metric_by_pupil",
    save=True,
):
    """
    Pupil–behaviour relationship plots.
    mode = 'metric_by_pupil': metric ~ pupil (bins on pupil axis)
    mode = 'pupil_by_metric': pupil ~ metric (bins on metric axis)
    """
    D = df.copy()
    if isinstance(pupil_cols, str):
        pupil_cols = (pupil_cols,)
    pupil_cols = [c for c in pupil_cols if c in D.columns]
    if not pupil_cols:
        print("No pupil columns found.")
        return
    main_pupil = pupil_cols[0]

    if regress_reward:
        D = residualize_pupil(D, pupil_cols, predictors=predictors, by="Session")
        pupil_cols = [f"{c}_clean" for c in pupil_cols if f"{c}_clean" in D.columns]
        if not pupil_cols:
            print("No cleaned pupil columns produced.")
            return
        main_pupil = pupil_cols[0]

    if groups is None:
        groups = {
            "Accuracy":  ["Correct_z", "Prec_z", "Bias_z", "TD_z", "PL_z"],
            "Speed":     ["RT_z", "SpeedMean_z"],
            "Difficulty": ["MorphFold"],
        }

    xvars = []
    for _, cols in groups.items():
        cols = [c for c in cols if c in D.columns]
        xvars.extend(cols)
    if not xvars:
        print("No x-vars present in DataFrame.")
        return

    n_cols = len(xvars)
    fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 2.7), sharey=False)
    axes = np.atleast_1d(axes)

    if mode == "pupil_by_metric":
        pc = main_pupil
        for c, xv in enumerate(xvars):
            ax = axes[c]
            label_x = pretty_label(xv)
            label_y = pretty_label(pc)
            if smooth:
                out = _smooth_bin_stats(D, bin_col=xv, val_col=pc, n_centers=smooth_centers,
                                        width_frac=smooth_width_frac, min_per_bin=min_per_bin,
                                        agg=agg, ci_q=ci_q)
                if out is None:
                    ax.set_title(f"{label_x} (n/a)", fontsize=9); ax.grid(True, linestyle=":", alpha=0.5); continue
                x, y, lo, hi = out
                ax.plot(x, y, "-"); ax.fill_between(x, lo, hi, alpha=0.3)
            else:
                out = _bin_stats(D, bin_col=xv, val_col=pc, q=q, min_per_bin=min_per_bin, equal_positions=equal_spacing_top)
                if out is None:
                    ax.set_title(f"{label_x} (n/a)", fontsize=9); ax.grid(True, linestyle=":", alpha=0.5); continue
                x, y, e, labels = out
                ax.errorbar(x, y, yerr=e, fmt="o-", capsize=3)
                if labels is not None and equal_spacing_top:
                    idx = np.arange(len(labels)); ax.set_xticks(idx)
                    ax.set_xticklabels([labels[i] if i % 2 == 0 else "" for i in idx])
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_xlabel(label_x)
            if c == 0: ax.set_ylabel(label_y)
            ax.set_title(label_x, fontsize=10)
    else:
        p_lab = pretty_label(main_pupil)
        for c, xv in enumerate(xvars):
            ax = axes[c]
            x_lab = pretty_label(xv)
            if smooth:
                out = _smooth_bin_stats(D, bin_col=main_pupil, val_col=xv, n_centers=smooth_centers,
                                        width_frac=smooth_width_frac, min_per_bin=min_per_bin,
                                        agg=agg, ci_q=ci_q)
                if out is None:
                    ax.set_title(f"{x_lab} (n/a)", fontsize=9); ax.grid(True, linestyle=":", alpha=0.5); continue
                p_centers, x_stat, x_lo, x_hi = out
                if bottom_pupil_axis == "y":
                    ax.plot(x_stat, p_centers, "-"); ax.fill_betweenx(p_centers, x_lo, x_hi, alpha=0.3)
                    ax.set_xlabel(x_lab)
                    if c == 0: ax.set_ylabel(p_lab)
                else:
                    ax.plot(p_centers, x_stat, "-"); ax.fill_between(p_centers, x_lo, x_hi, alpha=0.3)
                    ax.set_xlabel(p_lab)
                    if c == 0: ax.set_ylabel(x_lab)
            else:
                out = _bin_stats(D, bin_col=main_pupil, val_col=xv, q=q, min_per_bin=min_per_bin, equal_positions=False)
                if out is None:
                    ax.set_title(f"{x_lab} (n/a)", fontsize=9); ax.grid(True, linestyle=":", alpha=0.5); continue
                pupil_centers, x_mean, x_sem, _ = out
                if bottom_pupil_axis == "y":
                    ax.errorbar(x_mean, pupil_centers, xerr=x_sem, fmt="o-", capsize=3)
                    ax.set_xlabel(x_lab)
                    if c == 0: ax.set_ylabel(p_lab)
                else:
                    ax.errorbar(pupil_centers, x_mean, yerr=x_sem, fmt="o-", capsize=3)
                    ax.set_xlabel(p_lab)
                    if c == 0: ax.set_ylabel(x_lab)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_title(x_lab, fontsize=10)

    fig.suptitle(
        f"{species} — Pupil–behaviour "
        f"(residualized={'yes' if regress_reward else 'no'}, "
        f"smooth={'yes' if smooth else 'no'}, mode={mode})",
        y=1.05, fontsize=12
    )
    fig.tight_layout()
    if save:
        save_plot(f"pupil_relationships_{species}.png")


def _norm_bar(map_name, max_value, vmin=0.4):
    """Creates a normalized colormap and scalar mappable object."""
    norm = Normalize(vmin=vmin, vmax=max_value)
    cmap = plt.get_cmap(map_name)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    return cmap, mappable


def plot_running_paths(species_data, selected_indices, metric='Prec', color_map='flare_r'):
    """
    Plot example running paths for each species, colour-coded by *metric*.

    Parameters
    ----------
    species_data : dict
        Raw session DataFrames keyed by species name.
    selected_indices : dict
        {species: array of integer trial indices} — paths to highlight.
    metric : str
        Column name used for colour coding (default 'Prec').
    color_map : str
        Matplotlib colormap name (default 'flare_r').
    """
    # Session to use per species (0-indexed position in the session list)
    _session_idx = {'human': 1, 'mouse': 3, 'monkey': 3}

    cmap, mappable = _norm_bar(color_map, max_value=1.0)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for i, (species, data) in enumerate(species_data.items()):
        sesh = data[_session_idx[species]].dropna().reset_index(drop=True)

        distance = sesh['Distance']
        boxsize = sesh.attrs['Metadata']['BoxSize']
        precision_data = sesh[metric]

        correct = sesh['Correct'].astype(bool)
        precision_data = precision_data[correct].reset_index(drop=True)
        paths = sesh['Location'][correct].reset_index(drop=True)
        rt = sesh['RT'][correct].reset_index(drop=True)

        indices = selected_indices[species]
        selected_paths = [paths[idx] for idx in indices]
        selected_precision = [precision_data[idx] for idx in indices]
        selected_rt = [rt[idx] for idx in indices]
        selected_distance = [distance[idx] for idx in indices]

        for path, prec, rt_idx, dist in zip(selected_paths, selected_precision, selected_rt, selected_distance):
            x_near_targ = np.argmin(abs(path[:, 0] - dist - boxsize / 2))
            if species == 'mouse':
                y = path[:x_near_targ, 0] - path[:, 0][0]
                x = path[:x_near_targ, 1] - path[:, 1][0]
            else:
                y = path[:x_near_targ, 0]
                x = path[:x_near_targ, 1]

            axs[i].plot(x, y, color=cmap(mappable.norm(prec)))
            axs[i].plot(x[rt_idx], y[rt_idx], marker='o', zorder=3,
                        markerfacecolor='None', markeredgecolor='k')

        axs[i].set_xlim([-300, 300])
        axs[i].set_ylim([-20, 600] if species == 'mouse' else [-20, 450])
        axs[i].set_title(f"{species.capitalize()} (n={len(selected_paths)})")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(mappable, cax=cbar_ax, label=metric)
    save_plot(f"example_paths_{metric}.png")
    plt.show()


def fold_difficulty(morph):
    """Fold morphs so 50 is hardest; map to 0–50."""
    m = np.asarray(morph, float)
    return np.where(m > 50, 100 - m, m)


def collect_windows_for_transition(df, metric, window, from_state, to_state):
    """
    Collect z-scored folded difficulty windows around a specific transition
    (from_state -> to_state) for all sessions of one species.
    Returns: [n_transitions x (2*window+1)].
    """
    all_windows = []
    for sesh_id, sesh in df.groupby('Session'):
        sesh = sesh.sort_values('Trial') if 'Trial' in sesh.columns else sesh.copy()

        folded = fold_difficulty(sesh['MorphTarget'].values)
        mu, sd = folded.mean(), folded.std()
        zdiff = np.zeros_like(folded, dtype=float) if (sd == 0 or np.isnan(sd)) else (folded - mu) / sd

        states = sesh[metric].values
        n = len(states)
        if n < 2 * window + 2:
            continue

        for t in range(1, n):
            if states[t - 1] != from_state or states[t] != to_state:
                continue
            if t < window or t > n - 1 - window:
                continue
            w = zdiff[t - window:t + window + 1]
            if len(w) == 2 * window + 1:
                all_windows.append(w.astype(float))

    if not all_windows:
        return np.empty((0, 2 * window + 1), dtype=float)
    return np.vstack(all_windows)


def difficulty_psth_by_transition(species_data, metric='States_Speed',
                                   window=10, n_perm=5000,
                                   cmap_speed=None, cmap_accuracy=None):
    """
    PSTH of folded difficulty (z-score) around transitions, with permutation null
    and Bonferroni-corrected significance markers. Background shading shows from/to
    state colour; line is the observed mean ± SEM.
    """
    lags = np.arange(-window, window + 1)

    fig, axs = plt.subplots(2, len(species_data), figsize=(6.5, 4.2),
                            sharey=True, sharex=True)

    for col, species in enumerate(species_data):
        df = load_pickle('hmm_sessions', species)
        uniq_states = np.sort(df[metric].dropna().unique())
        if len(uniq_states) < 2:
            for row in range(2):
                axs[row, col].set_visible(False)
            continue

        s0, s1 = uniq_states[0], uniq_states[1]
        trans_pairs = [(s0, s1), (s1, s0)]

        if metric == 'States_Speed':
            name = {s0: 'slow', s1: 'fast'}
            colour = {
                s0: cmap_speed.colors[0] if cmap_speed else '0.6',
                s1: cmap_speed.colors[1] if cmap_speed else '0.9',
            }
        elif metric == 'States_Accuracy':
            name = {s0: 'incorrect', s1: 'correct'}
            colour = {
                s0: cmap_accuracy.colors[0] if cmap_accuracy else '0.6',
                s1: cmap_accuracy.colors[1] if cmap_accuracy else '0.9',
            }
        else:
            name = {s0: str(s0), s1: str(s1)}
            colour = {s0: '0.7', s1: '0.9'}

        def trans_label(fr, to):
            return f"{name[fr]} -> {name[to]}"

        for row, (fr, to) in enumerate(trans_pairs):
            ax = axs[row, col]
            windows = collect_windows_for_transition(df, metric, window, fr, to)
            n_tr = windows.shape[0]

            if n_tr == 0:
                ax.set_title(f"{species.capitalize()}: {trans_label(fr, to)} (n=0)", fontsize=8)
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
                ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
                ax.set_xlim(-window, window)
                ax.set_ylim(-1, 1)
                continue

            mean_psth = windows.mean(axis=0)
            sem_psth  = windows.std(axis=0, ddof=1) / np.sqrt(n_tr)

            # permutation null: shuffle time within each window
            null_means = np.empty((n_perm, windows.shape[1]), dtype=float)
            for i in range(n_perm):
                shuffled = np.apply_along_axis(np.random.permutation, 1, windows)
                null_means[i] = shuffled.mean(axis=0)

            lo, hi = np.percentile(null_means, [2.5, 97.5], axis=0)

            # per-lag p-values (two-sided), Bonferroni correction
            pvals = np.empty_like(mean_psth)
            for j in range(mean_psth.size):
                diffs = null_means[:, j] - null_means[:, j].mean()
                obs   = mean_psth[j]     - null_means[:, j].mean()
                pvals[j] = (np.sum(np.abs(diffs) >= np.abs(obs)) + 1) / (n_perm + 1)

            sig_mask = pvals < (0.05 / mean_psth.size)

            # background: from-state left, to-state right
            ax.axvspan(-window, 0, color=colour[fr], alpha=0.25, zorder=-3)
            ax.axvspan(0, window,  color=colour[to], alpha=0.25, zorder=-3)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, zorder=-2)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8, zorder=-2)

            # null band
            ax.fill_between(lags, lo, hi, color='gray', alpha=0.25)

            # observed mean ± SEM
            ax.fill_between(lags, mean_psth - sem_psth, mean_psth + sem_psth,
                            alpha=0.2, color='black')
            ax.plot(lags, mean_psth, color='black', linewidth=1.5)

            # significant lags
            if np.any(sig_mask):
                ax.scatter(lags[sig_mask], mean_psth[sig_mask], color='red', s=12, zorder=5)

            if col == 0:
                ax.set_ylabel('Folded difficulty (z-score)', fontsize=8)
            ax.set_xlim(-window, window)
            ax.set_ylim(-1, 1)
            ax.set_yticks([-1, 0, 1])
            ax.set_title(f"{species.capitalize()}: {trans_label(fr, to)} (n={n_tr})", fontsize=8)
            if row == 1:
                ax.set_xlabel('Trials relative to transition', fontsize=8)

    plt.tight_layout()
    save_plot(f"difficulty_psth_{metric}_zscore_transitions.png")
    plt.show()


def analyze_cluster_reliability(corr_matrix, n_clusters=2, verbose=True):
    """
    Silhouette-based cluster reliability for a correlation matrix.
    Distance = 1 - r (signed). Uses average-linkage hierarchical clustering.
    Per-metric breakdown: a = mean intra-cluster distance, b = mean inter-cluster distance,
    s = (b - a) / max(a, b).
    Returns (overall_score, clusters_dict, per_metric_list).
    """
    dist_matrix = 1 - corr_matrix
    dist_condensed = dist_matrix.values[np.triu_indices(len(dist_matrix), k=1)]
    Z = linkage(dist_condensed, method='average')
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
    sample_scores = silhouette_samples(dist_matrix, cluster_labels, metric='precomputed')

    metric_names = corr_matrix.columns.tolist()
    per_metric = []
    for i, name in enumerate(metric_names):
        own_cluster = cluster_labels[i]
        own_mask = cluster_labels == own_cluster
        other_mask = ~own_mask
        dists = dist_matrix.values[i]
        a = dists[own_mask & (np.arange(len(dists)) != i)].mean() if own_mask.sum() > 1 else 0.0
        b = dists[other_mask].mean()
        per_metric.append({'metric': name, 'cluster': own_cluster, 'a': a, 'b': b, 's': sample_scores[i]})

    if verbose:
        print(f"  {'Metric':<18} {'Cluster':>7}   {'a (intra)':>9}  {'b (inter)':>9}  {'s':>6}")
        for m in per_metric:
            print(f"  {m['metric']:<18} {m['cluster']:>7}   {m['a']:>9.3f}  {m['b']:>9.3f}  {m['s']:>6.3f}")

    clusters = {}
    for i, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(metric_names[i])

    return score, clusters, per_metric
