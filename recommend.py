import torch
from collections import defaultdict
import torch.nn.functional as F

'''
Gets the track names of the original tracks in the playlist
'''
def get_playlist_info(item):
    print('Playlist ID:', item['pid'])
    print('Playlist Length:', len(item['tracks']))
    
    # Get track names---artist
    original_tracks = []
    for i in range(len(item['tracks'])):
        name = item['tracks'][i]['track_name']+'---'+item['tracks'][i]['artist_name']
        original_tracks.append(name)
        
    # Get track uris
    seeds = []
    for i in item['tracks']:
        uri = i['track_uri'].split(':')[-1]
        seeds.append(uri)
        
    return item, original_tracks, seeds

'''
Creates dictionary of highest scored recommendation (of songs not in playlist) for each song in playlist
seeds: list of track uris from user's playlist
dgl_G: DGL Graph
z: embeddings generated from model
pred: predictor from model
feat_data: matrix of feature data
'''
def recommend(seeds, dgl_G, z, pred, neigh, feat_data, uri_map):

    listed = list(uri_map) #parse through uri map for uri --> integer

    score_dict = defaultdict(dict)
    for s in seeds:
        s = uri_map[s]
        _, candidates = dgl_G.out_edges(s, form='uv')
        s_embed = z[s].unsqueeze(dim=0)
        edge_embeds = [torch.cat([s_embed, z[c.item()].unsqueeze(dim=0)],1) for c in candidates]
        #print('Node Value:', s, 'Possible Recs:', len(edge_embeds))
        edge_embeds = torch.cat(edge_embeds, 0)
        scores = pred.W2(F.relu(pred.W1(edge_embeds))).squeeze(1)
        val = list(zip(candidates.detach().numpy(), scores.detach().numpy()))
        val.sort(key=lambda x:x[1], reverse=True)
        
        # Make sure the song is not already in the playlist
        # score_dict[s] = val[0]
        inc = 0
        while True and inc < len(val):
            if listed[val[inc][0]] not in seeds:
                score_dict[s] = val[inc][0]
                break
            if inc == (len(val) - 1):
                # If no co-occurence, use 5-NN based on features -- COLD START
                # print('Cold Start, Using Feature Data Instead')
                closest = neigh.kneighbors(feat_data[[s]], 25, return_distance=False)[0]
                for i in closest:
                    if listed[i] not in seeds:
                        score_dict[s] = i
                        break
                break
                    
            else:
                inc += 1
                
    # Get uris            
    uri_recs = []
    for i in score_dict.keys():
        cur_uri = listed[score_dict[i]]
        uri_recs.append(cur_uri)
        
    return uri_recs

def recommend_no_repeat(seeds, dgl_G, z, pred, neigh, feat_data, uri_map):

    listed = list(uri_map) #parse through uri map for uri --> integer

    score_dict = defaultdict(dict)
    uri_recs = []
    for s in seeds:
        s = uri_map[s]
        _, candidates = dgl_G.out_edges(s, form='uv')
        s_embed = z[s].unsqueeze(dim=0)
        edge_embeds = [torch.cat([s_embed, z[c.item()].unsqueeze(dim=0)],1) for c in candidates]
        #print('Node Value:', s, 'Possible Recs:', len(edge_embeds))
        edge_embeds = torch.cat(edge_embeds, 0)
        scores = pred.W2(F.relu(pred.W1(edge_embeds))).squeeze(1)
        val = list(zip(candidates.detach().numpy(), scores.detach().numpy()))
        val.sort(key=lambda x:x[1], reverse=True)
        
        
        # Make sure the song is not already in the playlist or recommendations
        inc = 0       
        while True and inc < len(val):
            cur_uri = listed[val[inc][0]]
            if cur_uri not in seeds and cur_uri not in uri_recs:
                score_dict[s] = val[inc][0]
                uri_recs.append(cur_uri)
                break
            
            if not set(candidates)^set(uri_recs) or inc == (len(val) - 1):
                # If no co-occurence, use 5-NN based on features -- COLD START
                # print('Cold Start, Using Feature Data Instead')
                closest = neigh.kneighbors(feat_data[[s]], 5, return_distance=False)[0]
                for i in closest:
                    if listed[i] not in seeds:
                        score_dict[s] = i
                        uri_recs.append(cur_uri)
                        break
                break
                    
            else:
                inc += 1     
    
    return uri_recs