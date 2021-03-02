from qy_Elliptic_dataset_GCN import *
tag = "GCN_GCN_skipFalse"
model = GCN2layer(166, [100], GCNConv, GCNConv, use_skip=False)
model.load_state_dict(torch.load(f'model_GCN_GCN_skipFalse.pkl'))
print(model)