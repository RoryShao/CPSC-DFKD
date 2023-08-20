import torch.nn as nn


class WRNAdapter(nn.Module):
    """Adapt the features of MobileNet to teacher"""
    def __init__(self, head='mlp', model=None, feat_in=256, feat_out=256):
        super(WRNAdapter, self).__init__()
        self.model = model
        if head == 'linear':
            self.head = nn.Linear(feat_in, feat_out)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_in, feat_in),
                nn.ReLU(inplace=True),
                nn.Linear(feat_in, feat_out)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x, out_feature=False):
        if out_feature:
            origin_fea, out = self.model(x, out_feature=out_feature)
            feature = self.head(origin_fea)
            return feature, out
        else:
            out = self.model(x, out_feature=out_feature)
            return out

