import dgl
import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np
import re

def load_features(feat_dir, normalize=True):
    # Dont create graph from scratch
    '''
    if create_graph_from_scratch:
        num_nodes = get_gpickle('./data/playlists/', 'Spotify Playlist', gpickle_dir, playlist_num)

        # Run spotify API script to create features
        print('Pulling spotify song data using SpotifyAPI')
        pull_audio_features(num_nodes)

        print('Building songset features csv')
        build_songset_csv(feat_dir, num_nodes)
    '''
    print('Loading feature data...')
    data = np.genfromtxt(feat_dir, delimiter=',', skip_header=True, dtype=str)
    data = data[np.argsort(data[:, 13])]
    features = np.array(np.delete(data[:,1:], [11, 12, 13, 14, 15], 1), dtype=float)
    if normalize:
        features = F.normalize(torch.Tensor(features), dim=0)
    uris = data[:, 14]
    uris = [re.sub('spotify:track:', '', uri) for uri in uris]
    uri_map = {n: i for i,n in enumerate(uris)}
    print('Feature data shape: ' + str(features.shape))

    return features, uri_map

def load_graph(gpickle_dir, uri_map):
    print('Loading graph data...')

    G = nx.read_gpickle(gpickle_dir)
    print('Graph Info:\n', nx.info(G))

    src, dest = [], []
    weights = []
    for e in G.edges.data('weight'):
        uri_u, uri_v, w = e
        u, v = uri_map[uri_u], uri_map[uri_v]
        src.append(u)
        dest.append(v)
        w = G[uri_u][uri_v]['weight']
        weights.append(w)
  
    #make double edges
    src, dest = torch.tensor(src), torch.tensor(dest)
    src, dest = torch.cat([src, dest]), torch.cat([dest, src])
    dgl_G = dgl.graph((src, dest), num_nodes=len(G.nodes))
    
    #store edge weights in graph
    weights = torch.FloatTensor(weights+weights)
    dgl_G.edata['weights'] = weights
    
    return dgl_G, weights