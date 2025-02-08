import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ssm
import superlet as slt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize   

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

def train_hmm (data, n_states, rand_seed=0):
    np.random.seed(rand_seed)
    model = ssm.HMM(n_states,data.shape[1], observations="gaussian", transitions='sticky',transition_kwargs=dict(kappa=50))
        
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

def train_hmm_twice (data1,data2):
    #train two HMM models with 2 states
    states_1, state_means_1 = train_hmm(data1,3)
    states_2, state_means_2 = train_hmm(data2,3)
    
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

def prepare_and_train_dataset(dummy_index, data, accuracy_metrics, speed_metrics, shuffle=True):
    training_data = []
    for sesh_dict in data:
        sesh = pd.DataFrame.from_dict(sesh_dict)  # Convert dictionary to dataframe
        sesh = sesh.copy()
        if shuffle:
            sesh = sesh.sample(frac=1).reset_index(drop=True)
            
        sesh = detrend_learning(sesh, ['Prec', 'RT', 'SpeedMean'])
        sesh = smoothing(session=sesh, window=5, metrics=['Correct', 'Bias', 'Prec', 'RT', 'SpeedMean'])
        sesh = scaling(session=sesh, scaler='standard', metrics=['~Correct_Raw', 'Session', 'Bias_Raw', 'MorphTarget', 'MorphDistractor', 'Outcome', 'TrialDuration'])
        training_data.append(sesh)  # Keep as dataframe for now

    training_data = concatenating(training_data)
    min_val = training_data[accuracy_metrics + speed_metrics].min().min()
    training_data = insert_padding(training_data, padding_value=min_val - 0.001)

    training_acc = training_data[accuracy_metrics].values
    training_sp = training_data[speed_metrics].values
    
    training_data = training_data[training_data['Session'] >= 0].reset_index(drop=True)
    # Add HMM states to the training data
    states, state_means, states_accuracy, states_speed = train_hmm_twice(training_acc, training_sp)
    training_data['States'] = states
    training_data['States_Accuracy'] = states_accuracy
    training_data['States_Speed'] = states_speed

    return training_data.columns.tolist(), training_data.values  # Return as a tuple of columns and values

def compute_superlets_per_sesh(data,sesh_id, fs, foi):
    power_per_sesh=[]
    for isesh in np.unique(sesh_id):
        sesh=data[sesh_id==isesh].copy()
        shuffled_power = slt.superlets(sesh[np.newaxis,:], fs, foi)
        power_per_sesh.append(shuffled_power)
    power_per_sesh=np.concatenate(power_per_sesh,axis=1)
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
        session[metric]=detrpoly
    return session

def smoothing (session, window, metrics=None):
    metrics=select_metrics(session,metrics)
    session=session.copy()   
    for metric in metrics:
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

    parent_path, _ = os.path.split(os.getcwd())
    # current_path = os.getcwd()
    # while True:
    #     parent_path, current_folder = os.path.split(current_path)
    #     if current_folder == 'code':
    #         parent_path = parent_path  # The parent of 'code' directory
    #         break
    #     if not parent_path or parent_path == current_path:
    #         raise FileNotFoundError("Could not find the 'code' directory in the path hierarchy.")
    #     current_path = parent_path
        
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

def save_pickle(data, filename,species,savepath=None):

    if not savepath:
        parent_path, _ = os.path.split(os.getcwd())
        savepath=os.path.join(parent_path,'data')
    if not os.path.isdir(savepath):
        os.makedirs(savepath)   
        
    file_path = os.path.join(savepath, f'data_{filename}_{species}.pkl') 
    
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {file_path}")
       
def load_pickle(filename,species,savepath=None):
    if not savepath:
        parent_path, _ = os.path.split(os.getcwd())
        savepath=os.path.join(parent_path,'data')
            
    file_path = os.path.join(savepath, f'data_{filename}_{species}.pkl') 

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {file_path}")
    return data

def save_plot(species, *subfolders, filename="plot.png"):
    """
    Save the current plot to a specified path, creating any necessary subfolders.

    :param savepath: The base path where the plot should be saved.
    :param subfolders: Any number of subfolders to be created under the base path.
    :param filename: The filename to save the plot as (default is "plot.png").
    """
    parent_path, _ = os.path.split(os.getcwd())
    
    savepath = os.path.join(parent_path,'plot',f'{species}') #r"//gs/home/glukhovai/behaviour/plot/{species}".format(species=species)
    
    if filename[-3:] == 'png' or filename[-3:] == 'svg':
        im_format=filename[-3:]
    
    full_savepath = os.path.join(savepath, *subfolders)
    
    # Create subdirectories if they don't exist
    os.makedirs(full_savepath, exist_ok=True)
    
    full_filepath = os.path.join(full_savepath, filename)
    
    plt.savefig(full_filepath,format=im_format, dpi=300)
    if im_format=='svg':
        filename=filename[:-3]+'png'
        full_filepath = os.path.join(full_savepath, filename)
        plt.savefig(full_filepath,format='png', dpi=300)   

def norm_bar(data, map_name, max_value):
    """Creates a normalized colormap and scalar mappable object."""
    norm = Normalize(vmin=0.4, vmax=max_value)
    cmap = plt.get_cmap(map_name)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    return cmap, mappable

def plot_selected_running_paths(species_data, selected_indices_dict, metric='Prec', color_map='flare_r'):
    """
    Plot running paths for different species with color-coded metrics using selected indices.
    """
    # Set up the figure and colormap
    max_value = 1.0
    cmap, mappable = norm_bar(None, color_map, max_value)
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, (species, data) in enumerate(species_data.items()):
        # Select appropriate session data
        if species == 'human':
            sesh = data[1] 
        if species == 'mouse':
            sesh = data[10] 
        if species == 'monkey':
            sesh = data[3] 
        sesh.dropna(inplace=True)
        sesh.reset_index(drop=True, inplace=True)
        
        # Extract relevant data
        distance = sesh['Distance']
        boxsize = sesh.attrs['Metadata']['BoxSize']
        precision_data = sesh[metric]
        
        # Filter paths based on correct trials
        selected = sesh['Correct'].astype(bool)
        precision_data = precision_data[selected].reset_index(drop=True)
        paths = sesh['Location'][selected].reset_index(drop=True)
        rt = sesh['RT'][selected].reset_index(drop=True)
        
        # Get indices for this species
        try:
            species_indices = np.load(f'selected_paths_{species}.npy', allow_pickle=True)
            selected_paths = [paths[idx] for idx in species_indices]
            selected_precision = [precision_data[idx] for idx in species_indices]
            selected_rt = [rt[idx] for idx in species_indices]
            selected_distance = [distance[idx] for idx in species_indices]
        except:
            print(f"No selected paths found for {species}")
            continue
            
        # Print diagnostic information
        print(f"{species}: plotting {len(selected_paths)} selected paths")
        print(f"{species} precision range: {min(selected_precision):.3f} to {max(selected_precision):.3f}")
        
        # Plot paths
        for ipath, (path, prec, rt_idx, dist) in enumerate(zip(selected_paths, selected_precision, selected_rt, selected_distance)):
            # Find point nearest to target
            x_near_targ = np.argmin(abs(path[:, 0] - dist - boxsize / 2))
            
            # Calculate coordinates based on species
            if species == 'mouse':
                y = path[:x_near_targ, 0] - path[:, 0][0]
                x = path[:x_near_targ, 1] - path[:, 1][0]
            else:
                y = path[:x_near_targ, 0]
                x = path[:x_near_targ, 1]
            
            # Plot path with normalized color
            normalized_value = mappable.norm(prec)
            axs[i].plot(x, y, color=cmap(normalized_value))
            
            # Plot reaction time point
            axs[i].plot(
                x[rt_idx],
                y[rt_idx],
                marker='o',
                zorder=3,
                markerfacecolor='None',
                markeredgecolor='k'
            )
        
        # Set plot limits and title
        axs[i].set_xlim([-300, 300])
        axs[i].set_ylim([-20, 600] if species == 'mouse' else [-20, 450])
        axs[i].set_title(f"{species.capitalize()} (n={len(selected_paths)})")
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(mappable, cax=cbar_ax, label=metric)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    save_plot('cross_species',project, filename=f"example_paths_{metric}.svg")
    plt.show()
    
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
    
def rle_in_seconds(inarray, durations, time_seconds=False):
    """Run length encoding with delay times between runs of the same value and trial durations.
    If time_seconds is True, returns dwell times and delays in seconds; otherwise in counts.
    
    Args:
        inarray (array-like): Input array of states.
        durations (array-like): Durations corresponding to each state in inarray.
        time_seconds (bool): Flag to determine output units (True for seconds, False for counts).
        
    Returns:
        tuple: (runlengths, startpositions, delaytimes, values) in specified unit (counts or seconds)
    """
    ia = np.asarray(inarray)  # force numpy
    durations = np.asarray(durations)  # ensure durations is numpy array
    n = len(ia)
    if n == 0 or len(durations) != n:
        return (None, None, None, None)  # Also ensure durations is the correct length
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal
        i = np.append(np.where(y), n - 1)  # must include last element position
        z = np.diff(np.append(-1, i))  # run lengths (indices for states)
        p = np.cumsum(np.append(0, z))[:-1]  # start positions
        values = ia[i]

        # Convert run lengths and delays to seconds if time_seconds is True
        if time_seconds:
            z = np.array([durations[p[j]:p[j] + z[j]].sum() for j in range(len(z))])  # dwell times in seconds
            d = np.zeros(len(z), dtype=np.float64)  # initialize delays in seconds
            last_end_times = {}
            for idx, val in enumerate(values):
                start_time = durations[:p[idx]].sum()  # start time of the current state run
                if val in last_end_times:
                    delay = start_time - last_end_times[val]
                    d[idx] = delay
                last_end_times[val] = start_time + z[idx]  # update last end time for the state
        else:
            # Keep z as counts and initialize delay times array with zeros
            d = np.zeros(len(z), dtype=int)
            last_end_positions = {}
            for idx, val in enumerate(values):
                if val in last_end_positions:
                    delay = p[idx] - last_end_positions[val]
                    d[idx] = delay
                last_end_positions[val] = p[idx] + z[idx]

        return (z, p, d, values)
    
def fold_morph_values(morph_value):
    return 100 - morph_value if morph_value > 50 else morph_value

def compute_null_distribution(diffs_array, num_permutations):
    num_trials, num_points = diffs_array.shape  # Get the number of transitions and trial points
    null_distributions = np.zeros((num_points, num_permutations))
    
    for perm in range(num_permutations):
        shuffled_array = np.apply_along_axis(np.random.permutation, 1, diffs_array)  # Shuffle along each trial
        null_distributions[:, perm] = np.mean(shuffled_array >= 0, axis=0) * 100  # Calculate % positive for shuffled
    
    return null_distributions

def psth_state_occurrence(states, z, p, v, state_1, state_2, window, gap=5):
    starts_state = p[v == state_1]
    ends_state = starts_state + z[v == state_1] - 1
    psth = np.zeros(window * 2 + gap)
    
    for ibl, start in enumerate(starts_state):
        end = ends_state[ibl]

        for itr in range(1,window+1):
            if end + itr < len(states):
                if states[end + itr] == state_2:
                    psth[window - 1  + gap + itr] += 1
            if start - itr >= 0:
                if states[start - itr] == state_2:

                    psth[window - itr] += 1      
    return psth

def create_transition_matrix(states, minus_diag=False, window=1):
    # state_column= 'State'
    # df = pd.DataFrame(states, columns=[state_column])
    # states = df[state_column].unique()
    # num_states = len(states)
    # transition_matrix = np.zeros((num_states, num_states))

    # for i, j in zip(df[state_column][:-1], df[state_column].shift(-1)[:-1]):
    #     if not np.isnan(j):
    #         transition_matrix[int(i), int(j)] += 1
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

def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return (z, p, ia[i])

def shuffle_blocks(states):
    lengths, _, values = rle(states)
    permutation = np.random.permutation(len(lengths))
    shuffled_lengths = lengths[permutation]
    shuffled_values = values[permutation]
    shuffled_states = np.empty_like(states)
    start = 0
    for length, value in zip(shuffled_lengths, shuffled_values):
        shuffled_states[start:start + length] = value
        start += length
    return shuffled_states

def plot_states_timeseries(session_data,colormap_states,species):
    """ Plots the states sequence for a given session and adds a legend with updated colormap access """
    session_id=session_data['Session']
    num_sessions=len(np.unique(session_id))
    
    fig, axs = plt.subplots(num_sessions, 1, figsize=(15, num_sessions), sharex=True)
    fig.suptitle(f'States timeline, {species}')
    sesh_lens=[]
    for isesh in np.unique(session_id):
        ax=axs[int(isesh)]
        session_states=session_data['States'][session_id==isesh].reset_index(drop=True).values
        sesh_lens.append(len(session_states))
        # Create a color map for states using the updated method
        unique_states = np.unique(session_states)
        num_states=len(unique_states)
        
        ax.imshow(session_states[np.newaxis, :], cmap=colormap_states, aspect='auto')#400
        ax.set_yticks([])
        ax.set_ylabel(f'sesh {isesh}')
        # Adding legend
        handles = [plt.Rectangle((0,0),1,1,color=colormap_states(i / num_states), label=f'State {i}') for i in unique_states]
        # Formatting the plot
        ax.set_yticks([])

    fig.legend(handles, unique_states, title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()


