import json
import numpy as np
import pprint
from numpy.ma.extras import unique
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import umap
import os
from sklearn.metrics import silhouette_score
from search_lcrs import LCR_ENTROPY_DETECTOR

def lcr_based_features(acc, protein_sequence, protein_lcr_info):
    alphabet = np.asarray(['D', 'K', 'I', 'Y', 'G', 'R', 'M', 'E', 'L', 'W', 'P', 'F', 'H', 'T', 'N', 'A', 'C', 'V', 'S', 'Q'])
    feature_names = ['Accession'] + [f'p({letter})' for letter in alphabet] + ['H', 'Length', 'Charge', 'Position', 'Charge Type', 'Lcr Type']
    lcr_features = pd.DataFrame(columns = feature_names)
    for i, lcr in enumerate(protein_lcr_info):
        lcr_length = lcr[2] - lcr[1] + 1
        lcr_type_length = len(lcr[3])
        if  lcr_length >= 5 and lcr_type_length <= 3:
            feature_vector = [acc_gn[acc]]
            # AA frequencies
            aa_frequencies = LCR_ENTROPY_DETECTOR.aa_freq(lcr[0])
            feature_vector[1:] = aa_frequencies.values()

            # Shannon Entropy
            feature_vector.append(LCR_ENTROPY_DETECTOR.shannon_entropy(aa_frequencies))

            # Length
            feature_vector.append(lcr[2] - lcr[1] + 1)

            # Charge
            feature_vector.append(sum(lcr[5]))

            # Position
            feature_vector.append((lcr[2] + lcr[1])/(2 * len(protein_sequence)))

            # Charge Type
            charge_type_encoder = {'zero': 0, 'negative' : 1, 'positive' : 2, 'positive-negative' : 3}
            feature_vector.append(charge_type_encoder[lcr[4]])

            # Lcr Type
            feature_vector.append(lcr[3])
            lcr_features.loc[i] = feature_vector
    return lcr_features

# Loading FASTA for analysis
project_dir = os.getcwd()
data_dir = f'{project_dir}/data/HPA/'
path_to_fasta = f'{data_dir}/nucleolus_proteins.fasta'

acc_fasta = {}
acc_gn = {}
with open(path_to_fasta) as F:
    sequences = F.read().split('>')
    for seq in sequences:
        split_position = (seq.find('|') + 1, seq.find('|', seq.find('|') + 1), seq.find('\n'), seq.find('GN=') + 3, seq.find('PE=') - 1)
        acc_fasta[seq[split_position[0]:split_position[1]]] = seq[split_position[2]:].replace('\n', '').replace('U', 'V') # иногда в sequence возникает буква U (в UniProt)
        acc_gn[seq[split_position[0]:split_position[1]]] = seq[split_position[3]: split_position[4]]
    del acc_fasta['']
    del acc_gn['']

# Loading lcr info
with open('/home/user/PC/Python/charge_metrics/data/output/lcr_info_all.json') as F:
    lcr_info = json.load(F)

# Preparing features for UMAP
lcr_types = []
for lcr_list in lcr_info.values():
    lcr_types += [lcr[3] for lcr in lcr_list]
unique_types = list(unique(lcr_types))

alphabet = ['D', 'K', 'I', 'Y', 'G', 'R', 'M', 'E', 'L', 'W', 'P', 'F', 'H', 'T', 'N', 'A', 'C', 'V', 'S', 'Q']
feature_names = ['Accession'] + [f'p({letter}).Mean' for letter in alphabet] + ['H.Mean','Length.Mean', 'Length.Std', 'Charge.Mean', 'Charge.Std'] + [f'LcrCounts in part {i}' for i in range(5)] + ['Zero lcr counts', 'Negative lcr counts', 'Positive lcr counts', 'Positive-Negative lcr counts'] + [str(type_) for type_ in unique_types]
protein_features = pd.DataFrame(columns = feature_names)
for acc, info in lcr_info.items():
    alphabet = np.asarray(['D', 'K', 'I', 'Y', 'G', 'R', 'M', 'E', 'L', 'W', 'P', 'F', 'H', 'T', 'N', 'A', 'C', 'V', 'S', 'Q'])
    lcr_features = lcr_based_features(acc, acc_fasta[acc], info)
    if len(lcr_features) > 1:
        feature_vector = [acc_gn[acc]]
        feature_vector[1:] = [lcr_features[lcr_features.columns[i]].mean() for i in range(1, 22)]
        feature_vector.append(lcr_features['H'].mean())
        feature_vector.append(lcr_features['Length'].mean())
        feature_vector.append(lcr_features['Length'].std())
        feature_vector.append(lcr_features['Charge'].mean())
        feature_vector.append(lcr_features['Charge'].std())

        bins = [0, .25, .50, .75, 1]
        counts, _ = np.histogram(lcr_features['Position'], bins=bins)
        feature_vector[len(feature_vector):] = counts
        feature_vector.append((lcr_features['Charge Type'] == 0).sum())
        feature_vector.append((lcr_features['Charge Type'] == 1).sum())
        feature_vector.append((lcr_features['Charge Type'] == 2).sum())
        feature_vector.append((lcr_features['Charge Type'] == 3).sum())

        feature_vector[len(feature_vector):] = [(lcr_features['Lcr Type'] == type).count() for type in unique_types]

        protein_features.loc[len(protein_features)] = feature_vector

# VarianceThreshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
protein_features_var = selector.fit_transform(protein_features.iloc[:, 1:])

# Standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(protein_features_var)

# UMAP
reducer = umap.UMAP(n_components = 2, n_neighbors = 10, random_state = 25)
embedding = reducer.fit_transform(features_scaled)

# Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(embedding)
score = silhouette_score(embedding, clusters)

# UMAP visualisation
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='Spectral', s=50, alpha=0.7)
plt.legend(handles=scatter.legend_elements()[0],
               labels=set(clusters),
               title="Clusters")
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

# Saving clusters info
clusters_dir = f'{project_dir}'
protein_clusters = {}
for i in set(clusters):
    proteins = []
    rows = []
    for k, cl in enumerate(clusters):
        if cl == i:
            proteins.append(protein_features.iloc[k, 0])
            rows.append(protein_features.iloc[k, :].to_list())
    cluster = pd.DataFrame(rows, columns=feature_names)
    cluster.to_excel(f'{project_dir}/data/clusters/cluster_{i}.xlsx', index = False, engine='openpyxl')
    protein_clusters[f'cluster {i}'] = proteins

#Statistical approaches
len_lcr = []
lcr_types = []
for _ in lcr_info.values():
    for lcr in _:
        len_lcr.append(lcr[2] - lcr[1] + 1)
        lcr_types.append(lcr[3])
unique_types = list(unique(lcr_types))

unique_types_counts = {}
for type in unique_types:
    unique_types_counts[type] = lcr_types.count(type) / len(lcr_types)
unique_types_counts = sorted(unique_types_counts.items(), key = lambda item: item[1], reverse=True)
unique_types_counts = dict(unique_types_counts[0:10])
pprint.pprint(lcr_info)
print('Unique lcr types : ', unique_types)
print('First 10 top hits : ', unique_types_counts)
plt.vlines(x=range(len(unique_types_counts)), ymin=0, ymax=unique_types_counts.values(), linewidth=.5, color='#678A90')
plt.hlines(y = .01, xmin = 0, xmax = 10)
plt.scatter(x=range(len(unique_types_counts)), y=unique_types_counts.values(), s=2, color='#69516B')
plt.xticks(range(len(unique_types_counts)), unique_types_counts)
plt.show()

# Visualisation LCR distribution
new_lcr_inf = {}
for acc, lcr_list in lcr_info.items():
    new_lcr_inf[acc] = [lcr for lcr in lcr_list if lcr[2] - lcr[1] + 1 >= 5]

k = 0
z = 0
fig, ax1 = plt.subplots(5, 2)

quantiles = [5, 10]
protein_lcr_localization = {}
for type_ in unique_types_counts.keys():
    q1_bucket, q2_bucket, q3_bucket, q4_bucket = [], [], [], []
    for seq, value in new_lcr_inf.items():
        for l in value:
            if l[3] == type_:
                len_= l[2] - l[1] + 1
                localization = (l[2] + l[1]) / (2 * len(acc_fasta[seq]))
                if len_ <= 5:
                    q1_bucket.append(round(localization, 3))
                elif 5 < len_ <= 10:
                    q2_bucket.append(round(localization, 3))
                elif len_ > 10:
                    q3_bucket.append(round(localization, 3))
    protein_lcr_localization[type_] = [q1_bucket, q2_bucket, q3_bucket]
    g_all = np.concatenate([q1_bucket, q2_bucket, q3_bucket])

    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

    c1, _ = np.histogram(q1_bucket, bins = bins)
    c2, _ = np.histogram(q2_bucket, bins = bins)
    c3, _ = np.histogram(q3_bucket, bins = bins)

    width = bins[1] - bins[0]
    positions = bins[:-1]

    ax1[k][z].bar(positions, c1, width=width, align='edge', label='group1', color ='#e4d9d6') #'#d6c7dd'
    ax1[k][z].bar(positions, c2, width=width, align='edge', label='group2', bottom=c1, color ='#f1a196' ) #'#caaac2'
    ax1[k][z].bar(positions, c3, width=width, align='edge', label='group3', bottom=c1+c2, color ='#ea806b' ) #'#d94c3e''

    ax2 = ax1[k][z].twinx()
    sns.kdeplot(g_all, ax = ax2, color='#0a0c0a', legend=True) #'#FC8B5E'

    kde_line = ax2.lines[0]
    kde_y = kde_line.get_ydata()
    kde_x = kde_line.get_xdata()

    x_zero_index = 0
    x_one_index = 0
    F = True
    for i_, x_ in enumerate(kde_x):
        if x_ > 0 and F == True:
            x_zero_index = i_
            F = False
        if x_ > 1 and F == False:
            x_one_index = i_
            break
    kde_line.set_xdata(kde_x[x_zero_index:x_one_index])
    kde_line.set_ydata(kde_y[x_zero_index:x_one_index])

    if k == 3 and z == 1:
        ax1[k][z].legend([f'<= 5', f'6 to 10', f'> 10'], loc = 'lower left', title='LCR length:', bbox_to_anchor=(1.25, 1), prop={'size' : 14})
    ax1[k][z].set_xlim([0,1])
    ax1[k][z].spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1[k][z].set_ylabel('Counts')
    ax1[k][z].set_xlabel('Protein sequence, %')
    ax1[k][z].set_title(f'LCR type : {type_}', loc='left')
    k += 1
    if k == 5:
        z += 1
        k = 0
fig.subplots_adjust(left = .075, right = .8, top = .925, bottom=0.1, hspace= .4, wspace=.325)
plt.show()
