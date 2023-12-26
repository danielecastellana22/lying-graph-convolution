from .baselines import BaseDGN
from convs import LyingConv, SparseLyingConv


class LyingGCN(BaseDGN):
    def __init__(self, **kwargs):
        super(LyingGCN, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        conv = LyingConv(**params)
        return conv, conv.out_channels

    def __extract_conv_results__(self, conv_results):
        return conv_results


class SparseLyingGCN(BaseDGN):
    def __init__(self, **kwargs):
        super(SparseLyingGCN, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        conv = SparseLyingConv(**params)
        return conv, conv.out_channels

    def __extract_conv_results__(self, conv_results):
        return conv_results
