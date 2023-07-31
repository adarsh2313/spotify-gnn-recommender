import pandas as pd
import json
import tqdm

def build_songset_csv(feat_dir, num_nodes):
    f = open('/home/adarsh/projects/spotify-graphsage/data/spotify_scrape/songset0.json')
    data = json.load(f)
    f.close
    songset = pd.DataFrame(data['audio_features'])

    for i in tqdm.tqdm(range(1, num_nodes // 100 + 1)):
        f = open(f'/home/adarsh/projects/spotify-graphsage/data/spotify_scrape/songset{i}.json')
        data = json.load(f)
        f.close

        data = pd.DataFrame(data['audio_features'])
        songset = pd.concat([songset, data], ignore_index=True)

    songset_trim = songset.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url'])

    print(songset_trim.shape)
    songset.to_csv(feat_dir)