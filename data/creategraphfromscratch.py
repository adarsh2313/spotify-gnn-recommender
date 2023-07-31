from operator import xor
import pandas as pd
import networkx as nx
import torch
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

def load_data(path, dataset_name, playlist_num):
    
    G = nx.Graph(name = 'G')
    n = playlist_num
    count = 0
    print('Loading {} playlists'.format(n))
    for i in range(999, n, 1000):
        with open(path + 'mpd.slice.{}-{}.json'.format(i - 999, i)) as f:
            data = json.load(f)
            playlists = data['playlists']
            count += create_nodes_edges(G, playlists)

    print('Playlist co-occurences: ' + str(count))
    print('Graph Info:\n', nx.info(G))
    nodes = G.nodes
    dic = {'track_uri': nodes}  
        
    df = pd.DataFrame(dic)
    df.to_csv('./data/songs.csv') 

    return G

# Create graph nodes and edges
def create_nodes_edges(G, playlists):
    count = 0
    for p in playlists:
        tracks = p['tracks']

        # keep track of pairs in current playlist
        seen = set()

        for i in range(len(tracks)): # create nodes
            first = tracks[i]['track_uri'][14:]

            for j in range(i + 1, len(tracks)):
                second = tracks[j]['track_uri'][14:]

                if first != second: # create edges
                    # Ensures co-occurence increased only if edge pair is found in new playlist
                    if not G.has_edge(first, second):
                        G.add_edge(first, second, weight=1)
                        seen.add(frozenset([first, second]))
                    elif frozenset([first, second]) not in seen:
                        G[first][second]['weight'] += 1
                        seen.add(frozenset([first, second]))
                        count += 1

    return count

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def frac_mat_power(m, n):
    evals, evecs = torch.eig (m, eigenvectors = True)
    evals = evals[:, 0]                               
    mchk = torch.matmul (evecs, torch.matmul (torch.diag (evals), torch.inverse (evecs)))
    mchk - m                                          
    evpow = evals**(n)
    mpow = torch.matmul (evecs, torch.matmul (torch.diag (evpow), torch.inverse (evecs)))
    return mpow