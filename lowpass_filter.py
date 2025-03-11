import json, os
import statistics
import numpy as np
import matplotlib.pyplot as plt
from search_lcrs import SIGNAL_PREPROCESSING
from sklearn.metrics import root_mean_squared_error
from search_lcrs import LCR_ENTROPY_DETECTOR


class PROTEIN_BLOCKINESS(SIGNAL_PREPROCESSING):
    def __init__(self, name, sequence, lcr_info, RMSE): #sampling_frequency
        self.name = name
        self.sequence = sequence
        self.lcr_info = lcr_info
        self.RMSE = RMSE
    @staticmethod
    def ideal_low_pass_filter(charge_signal, cutoff_frequency):
        signal_length = len(charge_signal)

        # Fourier Transform
        spectrum = np.fft.fft(charge_signal)
        freqs = np.fft.fftfreq(signal_length, 1/5000)
        mask = np.abs(freqs) <= cutoff_frequency

        # Ideal Low Pass Filtering
        filtered_spectrum = spectrum * mask
        return np.real(np.fft.ifft(filtered_spectrum))

    def get_lcr_signal(self):
        lcr_signal = np.zeros(len(self.sequence))
        for lcr in self.lcr_info:
            lcr_signal[lcr[1]:lcr[2] + 1] = lcr[5]
        self.lcr_charge_signal = lcr_signal
    def low_pass_filtering(self):
        signal_length = len(self.charge_signal)
        cutoff_frequency, step = signal_length, signal_length // 4

        flag = True
        while flag:
            seq_filtered_signal = self.ideal_low_pass_filter(self.charge_signal, cutoff_frequency)
            lcr_filtered_signal = self.ideal_low_pass_filter(self.lcr_charge_signal, cutoff_frequency)

            seq_filtered_parts = [seq_filtered_signal[lcr[1]: lcr[2] + 1] for lcr in self.lcr_info]
            lcr_filtered_parts = [lcr_filtered_signal[lcr[1]: lcr[2] + 1] for lcr in self.lcr_info]

            rmse = [root_mean_squared_error(lcr_part, seq_part) for lcr_part, seq_part in zip(lcr_filtered_parts, seq_filtered_parts)]
            rmse = statistics.mean(rmse)
            self.seq_filtered_signal = seq_filtered_signal
            self.lcr_filtered_signal = lcr_filtered_signal

            if rmse < self.RMSE and step not in [0, 1]:
                cutoff_frequency += step
                step //= 2
            elif rmse > self.RMSE:
                cutoff_frequency -= step
            elif step in [0, 1]:
                flag = False

    def ncpr_based_thresholding(self):
        self.NCPR = np.abs(sum(self.charge_signal)/len(self.charge_signal))
        self.seq_threshold_signal = self.threshold_filter(self.seq_filtered_signal, self.NCPR, -self.NCPR)

    def charge_blocks(self):
        self.blocks = []
        SP = 0
        for i in range(len(self.seq_threshold_signal) - 1):
            if self.seq_threshold_signal[i + 1] != self.seq_threshold_signal[i] or i == len(self.seq_filtered_signal) - 2:
                if self.seq_threshold_signal[i] == 0:
                    EP = i + 1
                    block_charge_seq = [0]*(EP - SP + 1)
                    self.blocks.append([SP, EP, block_charge_seq])
                    SP = i + 1
                else:
                    EP = i
                    if EP - SP != 0:
                        block_charge = sum(self.charge_signal[SP: EP + 1])/(EP - SP + 1)
                        block_charge_seq = [block_charge]*(EP - SP + 1)
                        self.blocks.append([SP, EP, block_charge_seq])
                    else:
                        self.blocks.append([SP, EP, [0]*(EP - SP + 1)])
                    SP = i if self.seq_threshold_signal[i + 1] == 0 else i
    def visualisation(self):
        self.blocks_X = []
        self.blocks_Y = []
        for bl in self.blocks:
            self.blocks_Y += bl[2]
            self.blocks_X += list(range(bl[0], bl[1] + 1))

        fig, ax = plt.subplots(2, 1)
        for i in range(3):
            ax[0].plot(range(len(self.seq_filtered_signal)), self.seq_filtered_signal, linewidth=.5, color='tomato')
            ax[0].hlines(0, 0, len(self.seq_filtered_signal), linewidth=.5, color='dimgrey')
            ax[0].plot(range(len(self.lcr_filtered_signal)), self.lcr_filtered_signal, linewidth=.5, color='violet')
            ax[0].set_ylabel('Charge')
            ax[0].legend(['Sequence NCPR', 'Lcr NCPR'])

            ax[1].plot(self.blocks_X, self.blocks_Y, color='darkviolet', linewidth=.5)
            ax[1].hlines(0, 0, len(self.seq_threshold_signal), color='grey', linewidth=.5)
            ax[1].hlines(self.NCPR, 0, len(self.seq_threshold_signal), color='dimgrey', linestyles='--', linewidth=.5)
            ax[1].hlines(-self.NCPR, 0, len(self.seq_threshold_signal), color='dimgrey', linestyles='--', linewidth=.5)
            ax[1].set_xlabel('Aminoacid Position')
            ax[1].set_ylabel('Charge')
            ax[1].legend(['Charge Blockiness'])

            ax[0].set_title(f'{self.name}')
        plt.show()
        plt.close()

    def fit(self):
        # Get lcr and sequence signal
        self.get_lcr_signal()
        self.charge_signal = self.seq_charge_generator(self.sequence)

        # Low pass filtering
        self.low_pass_filtering()

        # Threshold filtering
        self.ncpr_based_thresholding()

        # Get charge blockiness
        self.charge_blocks()

        # Results visualisation
        self.visualisation()

if __name__ == '__main__':
    # Loading FASTA for analysis
    project_dir = os.getcwd()
    output_dir = f'{project_dir}/data/output/' # Use your own output dir
    data_dir = f'{project_dir}/data/HPA/'
    path_to_fasta = f'{data_dir}/nucleolus_proteins.fasta'  # Use your own path to .fasta

    acc_fasta = {}
    with open(path_to_fasta) as F:
        sequences = F.read().split('>')
        for seq in sequences:
            split_position = (seq.find('|') + 1, seq.find('|', seq.find('|') + 1), seq.find('\n'))
            acc_fasta[seq[split_position[0]:split_position[1]]] = seq[split_position[2]:].replace('\n', '').replace('U', 'V') # иногда в sequence возникает буква U (в UniProt)
        del acc_fasta['']

    # Load lcr info
    lcr_info_dir = f'{project_dir}/data/output/'
    with open(f'{lcr_info_dir}/lcr_info_all.json') as F:
        lcr_info = json.load(F)

    # Fit and dump
    charge_blockiness_info = {}
    len_list = []
    charge_list = []
    len_charge = []
    for acc, current_lcr in lcr_info.items():
        FILTER = PROTEIN_BLOCKINESS(acc, acc_fasta[acc], current_lcr, .1)
        FILTER.fit()
        charge_blockiness_info[acc] = {'Sequence Charge Signal' : FILTER.charge_signal, 'Lcr Charge Signal' : FILTER.lcr_charge_signal.tolist(), 'Sequence Filtered Signal' : FILTER.seq_filtered_signal.tolist(), 'Lcr Filtered Signal' : FILTER.lcr_filtered_signal.tolist(), 'Seq Threshold Signal' : FILTER.seq_threshold_signal.tolist(), 'Charge Blocks' : [FILTER.blocks_X, FILTER.blocks_Y]}
    with open(f'{output_dir}/charge_blockiness_info_all.json', 'wt') as F:
        json.dump(charge_blockiness_info, F)