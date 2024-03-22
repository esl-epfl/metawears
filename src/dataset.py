import json
import os
import pickle
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import stft, resample
from scipy.signal import filtfilt, butter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
from utils.parser_util import get_parser

GLOBAL_INFO = {}

siena_info_df = pd.read_json('../input/Siena_info.json')
args = get_parser().parse_args()


def search_walk(info):
    searched_list = []
    root_path = info.get('path')
    extensions = info.get('extensions')

    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in extensions:
                list_file = ('%s/%s' % (path, filename))
                searched_list.append(list_file)

    return searched_list


class TUHDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.file_length = len(self.file_list)
        self.transform = transform

    def __len__(self):
        return self.file_length

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            data_pkl = pickle.load(f)
            signals = np.asarray(data_pkl['STFT'])

            # signals = np.reshape(signals, (-1, signals.shape[2]))
            # signals = self.transform(signals)

            label = data_pkl['label']
            label = 0. if label == "bckg" else 1.
        return signals, label


def get_data_loader(batch_size, save_dir=args.TUSZ_data_dir):
    file_dir = {'train': os.path.join(save_dir, 'task-binary_datatype-train_STFT'),
                'val': os.path.join(save_dir, 'task-binary_datatype-dev_STFT'),
                'test': os.path.join(save_dir, 'task-binary_datatype-eval_STFT')}
    file_lists = {'train': {'bckg': [], 'seiz': []}, 'val': {'bckg': [], 'seiz': []}, 'test': {'bckg': [], 'seiz': []}}

    for dirname in file_dir.keys():
        filenames = os.listdir(file_dir[dirname])
        for filename in filenames:
            if 'bckg' in filename:
                file_lists[dirname]['bckg'].append(os.path.join(file_dir[dirname], filename))
            elif 'seiz' in filename:
                file_lists[dirname]['seiz'].append(os.path.join(file_dir[dirname], filename))
            else:
                print('------------------------  error  ------------------------')
                exit(-1)

    print('--------------------  file_lists  --------------------')
    for dirname in file_lists.keys():
        print('--------------------  {}'.format(dirname))
        for classname in file_lists[dirname].keys():
            print('{} num: {}'.format(classname, len(file_lists[dirname][classname])))

    train_data = file_lists['train']['bckg'] + file_lists['train']['seiz'] * \
                 int(len(file_lists['train']['bckg']) / len(file_lists['train']['seiz']))
    non_seizure_labels = np.zeros(len(file_lists['train']['bckg']))
    seizure_labels = np.ones(len(file_lists['train']['seiz']) *
                             int(len(file_lists['train']['bckg']) / len(file_lists['train']['seiz'])))
    train_label = np.concatenate((non_seizure_labels, seizure_labels))
    print('len(train_data): {}'.format(len(train_data)))

    val_data = file_lists['val']['bckg'] + file_lists['val']['seiz']
    test_data = file_lists['test']['bckg'] + file_lists['test']['seiz']

    val_label = np.concatenate((np.zeros(len(file_lists['val']['bckg'])),
                                np.ones(len(file_lists['val']['seiz']))))
    test_label = np.concatenate((np.zeros(len(file_lists['test']['bckg'])),
                                 np.ones(len(file_lists['test']['seiz']))))

    print('len(val_data): {}'.format(len(val_data)))
    print('len(test_data): {}'.format(len(test_data)))

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_data = TUHDataset(train_data, transform=train_transforms)
    val_data = TUHDataset(val_data, transform=val_transforms)
    test_data = TUHDataset(test_data, transform=test_transforms)

    return train_data, val_data, test_data, train_label, val_label, test_label


def get_data_loader_siena(batch_size, patient_ids, save_dir=args.siena_data_dir):
    file_dir = os.path.join(save_dir, 'task-binary_datatype-eval_STFT')

    file_lists = {'bckg': [], 'seiz': []}

    filenames = os.listdir(file_dir)
    for filename in filenames:
        patient = int(filename[2:4])
        if patient not in patient_ids:
            continue
        if 'bckg' in filename:
            file_lists['bckg'].append(os.path.join(file_dir, filename))
        elif 'seiz' in filename:
            file_lists['seiz'].append(os.path.join(file_dir, filename))
        else:
            print('------------------------  error  ------------------------')
            exit(-1)

    test_data = file_lists['bckg'] + file_lists['seiz']

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_data = TUHDataset(test_data, transform=test_transforms)

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=6)

    return test_loader


def get_data_loader_siena_finetune(patient_ids, save_dir=args.siena_data_dir):
    file_dir = os.path.join(save_dir, 'task-binary_datatype-eval_STFT')

    file_lists = {'bckg': [], 'seiz': []}

    filenames = os.listdir(file_dir)
    for filename in filenames:
        patient = int(filename[2:4])
        if patient not in patient_ids:
            continue
        if 'bckg' in filename:
            file_lists['bckg'].append(os.path.join(file_dir, filename))
        elif 'seiz' in filename:
            file_lists['seiz'].append(os.path.join(file_dir, filename))
        else:
            print('------------------------  error  ------------------------')
            exit(-1)

    seiz_len = len(file_lists['seiz'])
    target_len = 10
    ratio = 10 / seiz_len
    if seiz_len > target_len:
        train_data = file_lists['bckg'] + file_lists['seiz']
    else:
        train_data = file_lists['bckg'] + file_lists['seiz'] * (1 + target_len // seiz_len)
    non_seizure_labels = np.zeros(len(file_lists['bckg']))

    if seiz_len > target_len:
        seizure_labels = np.ones(len(file_lists['seiz']))
    else:
        seizure_labels = np.ones(len(file_lists['seiz'] * (1 + target_len // seiz_len)))
    train_label = np.concatenate((non_seizure_labels, seizure_labels))

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_data = TUHDataset(train_data, transform=train_transforms)

    return train_data, train_label


def _get_sample_frequency(signal_header):
    # Temporary conditional assignment while we deprecate 'sample_rate' as a channel attribute
    # in favor of 'sample_frequency', supporting the use of either to give
    # users time to switch to the new interface.
    return (signal_header['sample_rate']
            if signal_header.get('sample_frequency') is None
            else signal_header['sample_frequency'])


def read_edf(edf_file, ch_nrs=None, ch_names=None, digital=False, verbose=False):
    """
    Convenience function for reading EDF+/BDF data with pyedflib.
    Will load the edf and return the signals, the headers of the signals
    and the header of the EDF. If all signals have the same sample frequency
    will return a numpy array, else a list with the individual signals
    Parameters
    ----------
    edf_file : str
        link to an edf file.
    ch_nrs : list of int, optional
        The indices of the channels to read. The default is None.
    ch_names : list of str, optional
        The names of channels to read. The default is None.
    digital : bool, optional
        will return the signals as digital values (ADC). The default is False.
    verbose : bool, optional
        Print progress bar while loading or not. The default is False.
    Returns
    -------
    signals : np.ndarray or list
        the signals of the chosen channels contained in the EDF.
    signal_headers : list
        one signal header for each channel in the EDF.
    header : dict
        the main header of the EDF file containing meta information.
    """
    assert (ch_nrs is None) or (ch_names is None), \
        'names xor numbers should be supplied'
    if ch_nrs is not None and not isinstance(ch_nrs, list): ch_nrs = [ch_nrs]
    if ch_names is not None and \
            not isinstance(ch_names, list): ch_names = [ch_names]

    with pyedflib.EdfReader(edf_file) as f:
        # see which channels we want to load
        available_chs = [ch.upper() for ch in f.getSignalLabels()]
        n_chrs = f.signals_in_file

        # find out which number corresponds to which channel
        if ch_names is not None:
            ch_nrs = []
            for ch in ch_names:
                if not ch.upper() in available_chs:
                    print('will be ignored.')
                else:
                    ch_nrs.append(available_chs.index(ch.upper()))

        # if there ch_nrs is not given, load all channels

        if ch_nrs is None:  # no numbers means we load all
            ch_nrs = range(n_chrs)

        # convert negative numbers into positives
        ch_nrs = [n_chrs + ch if ch < 0 else ch for ch in ch_nrs]

        # load headers, signal information and
        header = f.getHeader()
        signal_headers = [f.getSignalHeaders()[c] for c in ch_nrs]

        # add annotations to header
        annotations = f.readAnnotations()
        annotations = [[s, d, a] for s, d, a in zip(*annotations)]
        header['annotations'] = annotations

        signals = []
        for i, c in enumerate(tqdm(ch_nrs, desc='Reading Channels',
                                   disable=not verbose)):
            signal = f.readSignal(c, digital=digital)
            signals.append(signal)

        # we can only return a np.array if all signals have the same samplefreq
        sfreqs = [_get_sample_frequency(shead) for shead in signal_headers]
        all_sfreq_same = sfreqs[1:] == sfreqs[:-1]
        if all_sfreq_same:
            dtype = np.int32 if digital else float
            signals = np.array(signals, dtype=dtype)

    assert len(signals) == len(signal_headers), 'Something went wrong, lengths' \
                                                ' of headers is not length of signals'
    del f
    return signals, signal_headers, header


def generate_lead_wise_data(edf_file):
    filename = '/'.join(edf_file.split('/')[-2:])
    signals, signal_headers, header = read_edf(edf_file)
    file_info = siena_info_df.loc[filename]
    fs = file_info['sampling_frequency']
    length = file_info['length']
    labels = file_info['labels']
    num_target_samples = int(length * GLOBAL_INFO['sample_rate'])
    signal_list = []
    disease_labels = {0: 'bckg', 1: 'seiz'}
    for bipolar_channel in file_info['bipolar_montage']:
        x = bipolar_channel[0]
        y = bipolar_channel[1]

        if x == -1 or y == -1:
            signal_list.append(np.zeros(num_target_samples))
            print("Channel not found in ", filename)
            continue

        bipolar_signal = signals[x] - signals[y]
        # Define the band-pass filter parameters
        lowcut = 0.1  # Lower cutoff frequency in Hz
        highcut = 80  # Upper cutoff frequency in Hz
        order = 4  # Filter order (adjust as needed)

        # Calculate the normalized cutoff frequencies
        nyquist_freq = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq

        # Design and apply the band-pass filter
        b, a = butter(order, [low, high], btype='band')
        bipolar_signal_filtered = filtfilt(b, a, bipolar_signal)

        if fs != GLOBAL_INFO['sample_rate']:
            bipolar_signal_resampled = resample(bipolar_signal_filtered, num_target_samples)
            signal_list.append(bipolar_signal_resampled)
        else:
            signal_list.append(bipolar_signal_filtered)

    signal_list_ordered = np.asarray(signal_list)

    for i, label in enumerate(labels):
        slice_eeg = signal_list_ordered[:,
                    int(i * GLOBAL_INFO['slice_length']) * GLOBAL_INFO['sample_rate']:
                    int((i + 1) * GLOBAL_INFO['slice_length']) * GLOBAL_INFO['sample_rate']]

        with open("{}/{}_label_{}_index_{}.pkl".format(GLOBAL_INFO['save_directory'],
                                                       edf_file.split('/')[-1].split('.')[0],
                                                       disease_labels[label], i), 'wb') as f:
            pickle.dump({'signals': slice_eeg, 'patient id': edf_file.split('/')[-1].split('.')[0].split('_')[0],
                         'label': disease_labels[label]}, f)


def generate_STFT(pickle_file):
    save_directory = "{}/task-{}_datatype-{}_STFT".format(args.save_directory, args.task_type, args.data_type)

    nperseg = 256
    noverlap = 64
    sampling_rate = 256
    freq_resolution = 2
    nfft = sampling_rate * freq_resolution
    cutoff_freq = 80

    with open(pickle_file, 'rb') as f:
        data_pkl = pickle.load(f)
        signals = np.asarray(data_pkl['signals'])
        if signals.shape != (20, 3072):
            print("Error in shape: ", signals.shape)

        freqs, times, spec = stft(signals, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                  boundary=None, padded=False)

        spec = spec[:, :cutoff_freq*freq_resolution, :]
        amp = (np.log(np.abs(spec) + 1e-10)).astype(np.float32)

        label = data_pkl['label']
        with open("{}/{}.pkl".format(save_directory, pickle_file.split('/')[-1].split('.')[0]), 'wb') as out_f:
            pickle.dump({'STFT': amp, 'label': label}, out_f)


def run_multi_process(f, l: list, n_processes=1):
    n_processes = min(n_processes, len(l))
    print('processes num: {}'.format(n_processes))

    results = []
    pool = Pool(processes=n_processes)
    for r in tqdm(pool.imap_unordered(f, l), total=len(l), ncols=75):
        results.append(r)

    pool.close()
    pool.join()

    return results


def main(args):
    channel_list = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG C3', 'EEG C4', 'EEG CZ',
                    'EEG T3', 'EEG T4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG T5', 'EEG T6', 'EEG PZ', 'EEG FZ']

    save_directory = "{}/task-{}_datatype-{}".format(args.save_directory, args.task_type, args.data_type)
    if os.path.isdir(save_directory):
        # os.system("rm -r {}".format(save_directory))
        raise ValueError("The directory is already there!")
    os.system("mkdir -p {}".format(save_directory))

    data_directory = "{}".format(args.data_directory)

    if args.task_type == "binary":
        disease_labels = {'bckg': 0, 'seiz': 1}
    else:
        exit(-1)

    edf_list = search_walk({'path': data_directory, 'extensions': [".edf", ".EDF"]})

    GLOBAL_INFO['channel_list'] = channel_list
    GLOBAL_INFO['disease_labels'] = disease_labels
    GLOBAL_INFO['save_directory'] = save_directory
    GLOBAL_INFO['label_type'] = args.label_type
    GLOBAL_INFO['sample_rate'] = args.sample_rate
    GLOBAL_INFO['slice_length'] = args.slice_length
    # GLOBAL_INFO['disease_type'] = args.disease_type

    print("Number of EDF files: ", len(edf_list))
    for i in GLOBAL_INFO:
        print("{}: {}".format(i, GLOBAL_INFO[i]))
    with open(save_directory + '/preprocess_info.pickle', 'wb') as pkl:
        pickle.dump(GLOBAL_INFO, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    # for edf_file in tqdm(edf_list[:60]):
    #     generate_lead_wise_data(edf_file)
    run_multi_process(generate_lead_wise_data, edf_list, n_processes=6)


def make_STFT(args):
    save_directory = "{}/task-{}_datatype-{}_STFT".format(args.save_directory, args.task_type, args.data_type)
    if os.path.isdir(save_directory):
        # os.system("rm -r {}".format(save_directory))
        raise ValueError("The directory is already there!")
    os.system("mkdir -p {}".format(save_directory))

    data_directory = "{}/task-{}_datatype-{}".format(args.save_directory, args.task_type, args.data_type)
    pickle_list = []
    for pickle_file in os.listdir(data_directory):
        if pickle_file.endswith(".pkl"):
            pickle_list.append(os.path.join(data_directory, pickle_file))

    run_multi_process(generate_STFT, pickle_list, n_processes=6)
    # for pickle_file in tqdm(pickle_list[:1]):
    #     generate_STFT(pickle_file)


def save_validation_inference():
    batch_size = 100
    _, val_loader, test_loader = get_data_loader_siena(batch_size)
    validation_file_mask_dict = {}
    for idx, (data, label, file, mask) in enumerate(tqdm(val_loader, desc='Validation ')):
        file_mask_dict = dict(zip(file, mask))
        validation_file_mask_dict.update(file_mask_dict)

    # Save 'dict' to a file
    with open('validation_file_mask8_dict.pkl', 'wb') as f:
        pickle.dump(validation_file_mask_dict, f)


if __name__ == '__main__':
    make_STFT(args)
    # main(args)
    pass
