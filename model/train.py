from utils.utils import load_graph
from utils.utils import load_features
from train_script import train
import torch
import json

feat_dir = "../data/songset_features.csv"
scratch_pickle_dir = "../data"
feat_data, uri_map = load_features(feat_dir)
graph_dir = ("../data/graph_main.gpickle")
dgl_G, weights = load_graph(graph_dir, uri_map)

# Training the Model. GPU ~ 00:00:40. CPU ~ 00:53:00.
with open('../config/model-params.json') as fh:
            model_cfg = json.load(fh)
model, pred, measures = train(dgl_G, weights.to('cpu'), feat_data, cuda=False, feat_dim=13, emb_dim=10, test_data=False)

# Put everything on CPU
model = model.to('cpu')
pred = pred.to('cpu')

torch.save(model, 'main_1epoch_model.pt')
torch.save(pred, 'main_1epoch_pred.pt')