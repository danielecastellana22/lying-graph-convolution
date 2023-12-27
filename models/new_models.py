from .baselines import BaseDGN
from lying_convs import LyingGCNConv, LyingGCN2Conv


class LyingGCN(BaseDGN):
    def __init__(self, **kwargs):
        super(LyingGCN, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        conv = LyingGCNConv(**params)
        return conv, conv.out_channels

    def __extract_conv_results__(self, conv_results):
        return conv_results


class LyingGCN2(BaseDGN):
    def __init__(self, **kwargs):
        super(LyingGCN2, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        in_ch = params.pop('in_channels')
        out_ch = params.pop('out_channels')
        assert in_ch == out_ch
        params['channels'] = in_ch
        conv = LyingGCN2Conv(**params)
        return conv, conv.channels

    def __extract_conv_results__(self, conv_results):
        return conv_results
