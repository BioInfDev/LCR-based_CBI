import numpy as np
from math import log2
import os, math, json

class SIGNAL_PREPROCESSING:
    @staticmethod
    def threshold_filter(signal, threshold_up, threshold_low):
        _filter_threshold_ = np.where(signal > threshold_up, 1, np.where(signal < threshold_low, -1, 0))
        return _filter_threshold_
    @staticmethod
    def seq_charge_generator(seq):
        charge_aa = {'K': 1, 'R': 1, 'H': 1, 'E': -1, 'D': -1, 'S': 0, 'A': 0, 'V': 0, 'L': 0, 'I': 0, 'M': 0, 'P': 0,'G': 0, 'T': 0, 'C': 0, 'Q': 0, 'N': 0, 'F': 0, 'Y': 0, 'W': 0}
        return [charge_aa[aa] for aa in seq if aa in charge_aa.keys()]
    @staticmethod
    def signal_discretisation(charge_signal, sampling_frequency):
        n = math.floor(len(charge_signal) / (2 * math.pi / sampling_frequency))
        X_ = np.linspace(0, len(charge_signal) - 1, n, endpoint= True)
        Y_ = np.asarray([charge_signal[math.ceil(point)] for point in X_])
        return X_, Y_
    @staticmethod
    def sliding_subsequences(seq, window: int):
        upper_bound = len(seq) - window + 1
        return [seq[l_:l_ + window] for l_ in range(upper_bound)]

class LCR_DETECOR:
    @staticmethod
    def aa_freq(seq):
        '''Accepts the aa sequence and return dict {AminoAcid : Freq}'''
        aa_symbol = ['D', 'K', 'I', 'Y', 'G', 'R', 'M', 'E', 'L', 'W', 'P', 'F', 'H', 'T', 'N', 'A', 'C', 'V', 'S', 'Q']
        aa_freq = {'D': 0.0, 'K': 0.0, 'I': 0.0, 'Y': 0.0, 'G': 0.0, 'R': 0.0, 'M': 0.0, 'E': 0.0, 'L': 0.0, 'W': 0.0,
                   'P': 0.0, 'F': 0.0, 'H': 0.0, 'T': 0.0, 'N': 0.0, 'A': 0.0, 'C': 0.0, 'V': 0.0, 'S': 0.0, 'Q': 0.0}
        aa_counts = sum([seq.count(aa) for aa in aa_symbol])
        for aa in aa_symbol:
            aa_freq[aa] += seq.count(aa) / aa_counts
        return aa_freq
    @staticmethod
    def lcr_statistically_significant(n_bootstrap, observation_param, sample_params: []):
        n_ = n_bootstrap
        bootstrap_medians = []
        for _ in range(n_):
            sample = np.random.choice(sample_params, size=len(sample_params), replace=True)
            bootstrap_medians.append(np.median(sample))
        ci_lower = np.percentile(bootstrap_medians, 0.5)
        ci_upper = np.percentile(bootstrap_medians, 99.5)
        result = False if ci_lower <= observation_param <= ci_upper else True
        return result
    @staticmethod
    def pseudorandom_sequence_generator(length, aa_frequencies, sequence_rate):
        np.random.seed(10)
        aa_symbol = np.asarray(['D', 'K', 'I', 'Y', 'G', 'R', 'M', 'E', 'L', 'W', 'P', 'F', 'H', 'T', 'N', 'A', 'C', 'V', 'S', 'Q'])
        return [list(np.random.choice(a=aa_symbol, size=length, p=list(aa_frequencies.values()))) for i in   range(sequence_rate)]

class LCR_ENTROPY_DETECTOR(LCR_DETECOR):
    @classmethod
    def class_parametres_initialization(cls, aa_distribution):
        cls.length_rate = range(4,51)
        cls.sequence_rate = 10000
        cls.aa_dist = aa_distribution
        cls.alphabet = np.asarray(['D', 'K', 'I', 'Y', 'G', 'R', 'M', 'E', 'L', 'W', 'P', 'F', 'H', 'T', 'N', 'A', 'C', 'V', 'S', 'Q'])
        cls.negative_letters = ['S', 'E', 'D']
        cls.positive_letters = ['K', 'R', 'H']
        if os.path.exists(r'/home/user/PC/Python/charge_metrics/data/sample_entropies.json'):
            with open(r'/home/user/PC/Python/charge_metrics/data/sample_entropies.json', 'rt') as F:
                cls.sample_entropies = json.load(F)
        else:
            cls.sample_entropies_generator()
    @classmethod
    def sample_entropies_generator(cls):
        np.random.seed(10)
        cls.sample_entropies = {}
        for length in cls.length_rate:
            random_sequences = [list(np.random.choice(a=cls.alphabet, size=length, p=list(cls.aa_dist.values()))) for i in range(cls.sequence_rate)]
            sequences_frequencies = [LCR_ENTROPY_DETECTOR.aa_freq(seq_) for seq_ in random_sequences]
            cls.sample_entropies[length] = [LCR_ENTROPY_DETECTOR.shannon_entropy(freqs_) for freqs_ in sequences_frequencies]
        with open(r'/home/user/PC/Python/charge_metrics/data/sample_entropies.json', 'at') as F:
            json.dump(cls.sample_entropies, F)
    @staticmethod
    def shannon_entropy(aa_freq: {}):
        freq = list(aa_freq.values())
        entropy = sum([-(f * log2(f)) for f in freq if f != 0])
        return entropy
    @staticmethod
    def entropy_trimmer(lcr_sequence, start, end):
        #Left edge trimmer
        i = 1
        not_trimmed_sequence = lcr_sequence
        trimmed_sequence = lcr_sequence[i:]
        while True:
            not_trimmed_freqs = LCR_ENTROPY_DETECTOR.aa_freq(not_trimmed_sequence)
            trimmed_freqs = LCR_ENTROPY_DETECTOR.aa_freq(trimmed_sequence)
            not_trimmed_entropy = LCR_ENTROPY_DETECTOR.shannon_entropy(not_trimmed_freqs)
            trimmed_entropy = LCR_ENTROPY_DETECTOR.shannon_entropy(trimmed_freqs)
            if not_trimmed_entropy > trimmed_entropy:
                start += 1
                i += 1
                not_trimmed_sequence = trimmed_sequence
                trimmed_entropy = lcr_sequence[i:]
            else:
                break
        #Right edge trimmer
        i = len(lcr_sequence) - 1
        not_trimmed_sequence = lcr_sequence
        trimmed_sequence = lcr_sequence[:i]
        while True:
            not_trimmed_freqs = LCR_ENTROPY_DETECTOR.aa_freq(not_trimmed_sequence)
            trimmed_freqs = LCR_ENTROPY_DETECTOR.aa_freq(trimmed_sequence)
            not_trimmed_entropy = LCR_ENTROPY_DETECTOR.shannon_entropy(not_trimmed_freqs)
            trimmed_entropy = LCR_ENTROPY_DETECTOR.shannon_entropy(trimmed_freqs)
            if not_trimmed_entropy > trimmed_entropy:
                end -= 1
                i -= 1
                not_trimmed_sequence = trimmed_sequence
                trimmed_entropy = lcr_sequence[:i]
            else:
                break
        return start, end

    def __init__(self, protein_id, sequence_signal):
        self.protein_id = protein_id
        self.sequence = sequence_signal
        self.kernel = 3
        self.random_entropy = 1.4666430958181733 # iterations : 1'000'000 ; kernel : 3
        self.status = 'OK'

    def get_entropy_signal(self):
        subsequences = SIGNAL_PREPROCESSING.sliding_subsequences(self.sequence, self.kernel)
        subsequences_aa_frequencies = [LCR_DETECOR.aa_freq(seq_) for seq_ in subsequences]
        self.entropy_signal = [LCR_ENTROPY_DETECTOR.shannon_entropy(freq) for freq in subsequences_aa_frequencies]
    def entropy_signal_edge_recoverer(self):
        self.entropy_signal.append(self.entropy_signal[-1])
        self.entropy_signal.insert(0, self.entropy_signal[0])
        self.entropy_signal = np.asarray(self.entropy_signal)
    def get_lcr(self):
        self.lcr = []
        SP = 0
        F = False
        for i_, e_ in enumerate(self.entropy_signal):
            if e_ < self.random_entropy and F == False:
                F = True
                SP = i_ - 1 if i_ != 0 else i_
            elif e_ >= self.random_entropy and F == True:
                F = False
                EP = i_
                self.lcr.append([self.sequence[SP : EP + 1], SP, EP])
    def get_trimmed_lcr(self):
        for i, _ in enumerate(self.lcr):
            trimmed_start, trimmed_end = LCR_ENTROPY_DETECTOR.entropy_trimmer(_[0], _[1], _[2])
            trimmed_sequence = self.sequence[trimmed_start: trimmed_end + 1]
            updated_lcr = [trimmed_sequence, trimmed_start, trimmed_end]
            self.lcr[i] = updated_lcr
    def get_lcr_type(self):
        for i, _ in enumerate(self.lcr):
            aa_freq = LCR_DETECOR.aa_freq(_[0])
            lcr_type = ''
            for aa_ in LCR_ENTROPY_DETECTOR.alphabet:
                if aa_freq[aa_] != 0 and -log2(aa_freq[aa_]) < 2:
                    lcr_type += aa_
            if lcr_type == '':
                lcr_type = ''.join(set(list(self.lcr[i][0])))
                self.lcr[i].append(lcr_type)
            else:
                self.lcr[i].append(lcr_type)
    def get_lcr_signature(self):
        for i, _ in enumerate(self.lcr):
            signature = {'positive': LCR_ENTROPY_DETECTOR.positive_letters,
                         'negative': LCR_ENTROPY_DETECTOR.negative_letters}
            type_list = list(_[3])
            lcr_signature = []
            for sign, type in signature.items():
                for aa in type:
                    if aa in type_list:
                        lcr_signature.append(sign)
                        break
            if lcr_signature == []:
                lcr_signature.append('zero')
            elif 'positive' in lcr_signature and 'negative' in lcr_signature:
                lcr_signature = ['positive-negative']
            self.lcr[i].append(lcr_signature[0])
    def lcr_assembly(self):
        # Assembly of overlapping lcr
        i = 0
        while i < len(self.lcr) - 1:
            if self.lcr[i + 1][1] - self.lcr[i][2] == 0:
                self.lcr[i][2] = self.lcr[i + 1][2]
                self.lcr[i][0] = self.sequence[self.lcr[i][1] : self.lcr[i][2] + 1]
                self.lcr.remove(self.lcr[i + 1])
                i -= 1
            i += 1
        self.lcr = [lcr[:3] for lcr in self.lcr]
        self.get_lcr_type()
        self.get_lcr_signature()

        # Charge-based lcr assembly
        i = 0
        while i < len(self.lcr) - 1:
            if self.lcr[i + 1][1] - self.lcr[i][2] > 0 and self.lcr[i + 1][4] == self.lcr[i][4] and self.lcr[i][4] in ['negative', 'positive']:
                signature = self.lcr[i][4]
                bridge_sequence = list(self.sequence[self.lcr[i][2] + 1 : self.lcr[i+1][1]])
                bridge_signature = True if bridge_sequence == [] else False
                bridge_alphabet = LCR_ENTROPY_DETECTOR.negative_letters if signature == 'negative' else LCR_ENTROPY_DETECTOR.positive_letters
                for letter in bridge_sequence:
                    if letter in bridge_alphabet:
                        bridge_signature = True
                    else:
                        bridge_signature = False
                        break
                if bridge_signature == True:
                    self.lcr[i][2] = self.lcr[i + 1][2]
                    self.lcr[i][0] = self.sequence[self.lcr[i][1] : self.lcr[i][2] + 1]
                    self.lcr.remove(self.lcr[i + 1])
                    i -= 1
            i += 1
        self.lcr = [lcr[:3] for lcr in self.lcr]
        self.get_lcr_type()
        self.get_lcr_signature()
    def charge_based_lcr_expansion(self):
        for i in range(len(self.lcr) - 1):
            if self.lcr[i][4] in ['positive', 'negative']:
                charge_alphabet = LCR_ENTROPY_DETECTOR.negative_letters if self.lcr[i][4] == 'negative' else LCR_ENTROPY_DETECTOR.positive_letters
                SP, EP = self.lcr[i][1] , self.lcr[i][2]
                while True:
                    if self.sequence[SP - 1] in charge_alphabet:
                        SP = SP - 1
                    else:
                        break
                while True:
                    if self.sequence[EP + 1] in charge_alphabet:
                        EP = EP + 1
                    else:
                        break
                self.lcr[i][1], self.lcr[i][2] = SP, EP
                self.lcr[i][0] = self.sequence[SP : EP + 1]
        self.lcr = [lcr[:3] for lcr in self.lcr]
        self.get_lcr_type()
        self.get_lcr_signature()
    def get_charge_sequence_of_lcr(self):
        for i, lcr in enumerate(self.lcr):
            self.lcr[i].append(SIGNAL_PREPROCESSING.seq_charge_generator(lcr[0]))
    def get_lcr_info(self):
        # Get lcr sequences and borders
        self.get_lcr()
        self.get_trimmed_lcr() # First : trimming; Second : assembly otherwise resolution (delta) != 1
        self.get_lcr_type()
        self.get_lcr_signature()

        # Lcr assembly
        self.lcr_assembly()

        # Charge-based lcr expansion
        self.charge_based_lcr_expansion()

        # Filtering lcr by length
        #self.lcr = [lcr for lcr in self.lcr if lcr[2] - lcr[1] >= 5]

        self.get_charge_sequence_of_lcr()
    def fit(self):
        try:
            self.charge_signal = SIGNAL_PREPROCESSING.seq_charge_generator(self.sequence)
            self.get_entropy_signal()
            self.entropy_signal_edge_recoverer()
            self.get_lcr_info()
        except:
            self.status = 'error'
if __name__ == '__main__':
    # Nucleolar proteome fusion to obtain amino acid frequencies
    nucleoli_fasta = open('/home/user/PC/Python/charge_metrics/data/HPA/nucleolus_proteins.fasta').read().split('>')
    merged_fasta = ''
    for seq in nucleoli_fasta:
        split_position = (seq.find('|') + 1, seq.find('|', seq.find('|') + 1), seq.find('\n'))
        merged_fasta += seq[split_position[2]:].replace('\n', '')

    # Loading FASTA for analysis
    project_dir = os.getcwd()
    data_dir = f'{project_dir}/data/HPA/'
    path_to_fasta = f'{data_dir}/nucleolus_proteins.fasta' # Use your own path to .fasta

    fasta_sequences = {}
    with open(path_to_fasta) as F:
        sequences = F.read().split('>')
        for seq in sequences:
            split_position = (seq.find('|') + 1, seq.find('|', seq.find('|') + 1), seq.find('\n'))
            fasta_sequences[seq[split_position[0]:split_position[1]]] = seq[split_position[2]:].replace('\n', '').replace('U', 'V')
        del fasta_sequences['']

    #Global parameters
    proteome_aa_frequencies = LCR_DETECOR.aa_freq(merged_fasta)
    output_dir = f'{project_dir}/data/output/' # Use your own output dir
    #Search LCRs : Shannon entropy method
    LCR_ENTROPY_DETECTOR.class_parametres_initialization(proteome_aa_frequencies)
    error_ = []
    lcr_info = {}
    for name, seq in fasta_sequences.items():
        ENTROPY_DETECTOR = LCR_ENTROPY_DETECTOR(name, seq)
        ENTROPY_DETECTOR.fit()
        if ENTROPY_DETECTOR.status == 'OK':
            lcr_info[name] = ENTROPY_DETECTOR.lcr
        else:
            error_.append([ENTROPY_DETECTOR.protein_id, ENTROPY_DETECTOR.status])
    with open(f'{output_dir}/lcr_info_all.json', 'wt') as F:
        json.dump(lcr_info, F)
    with open(f'{output_dir}/log_lcr_info_all.json', 'wt') as F:
        json.dump(error_, F)