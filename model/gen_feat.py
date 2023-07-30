import sys
sys.path.insert(0,'/home/adarsh/projects/new-folder/utils')
from utils import load_features
feat_dir = "/home/adarsh/projects/new-folder/data/songset_features.csv"
feat_data, uri_map = load_features(feat_dir)
print('Done')
print(type(uri_map))