from parser_util import get_parser
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from few_shot.support_set_const import seizure_support_set, non_seizure_support_set
from few_shot.support_set_const import seizure_support_set_siena, non_seizure_support_set_siena

options = get_parser().parse_args()

root_dir = os.path.join(options.TUSZ_data_dir,"task-binary_datatype-train_STFT")
average_list = []
std_list = []
for filename in non_seizure_support_set:
    filepath = os.path.join(root_dir, filename + ".pkl")
    with open(filepath, 'rb') as f:
        plt.figure()
        data_pkl = pickle.load(f)
        signals = np.asarray(data_pkl['STFT'])
        average_list.append(np.mean(signals))
        std_list.append(np.std(signals))
        sns.heatmap(np.reshape(signals, (-1, 15)))
        plt.savefig('../../output/vis/non_seiz_{}'.format(filename))
        plt.close()

print("Average ", np.mean(average_list))
print("STD ", np.mean(std_list))