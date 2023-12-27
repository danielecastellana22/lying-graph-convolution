import torch as th
import torch.nn as nn
import torch.nn.functional as F
from training import end_to_end_training


class BaseDGN(nn.Module):

    def __init__(self, n_layers, num_in_channels, num_hidden_channels, num_out_channels,
                 p_dropout, p_input_dropout, act_fun, skip_connections=False,
                 classifier_at_each_layer=False, **init_conv_params):
        super(BaseDGN, self).__init__()
        self.layers_list = nn.ModuleList()

        self.skip_connections = skip_connections
        self.p_dropout = p_dropout
        self.p_input_dropout = p_input_dropout
        self.input_lin = nn.Linear(num_in_channels, num_hidden_channels)

        assert n_layers > 0
        in_size = num_hidden_channels
        out_size_list = []

        for i in range(n_layers):
            layer, out_size = self.__init_conv__(in_channels=in_size,
                                                 out_channels=num_hidden_channels,
                                                 layer=i+1,
                                                 **init_conv_params)
            self.layers_list.append(layer)

            out_size_list.append(out_size)
            in_size = out_size

        self.classifier_at_each_layer = classifier_at_each_layer
        if classifier_at_each_layer:
            self.classifiers_list = nn.ModuleList([nn.Linear(o, num_out_channels) for o in out_size_list])
        else:
            self.classifiers_list = nn.ModuleList([nn.Linear(out_size_list[-1], num_out_channels)])

        if act_fun == 'tanh':
            self.act_fun = F.tanh
        elif act_fun == 'relu':
            self.act_fun = F.relu
        elif act_fun == 'elu':
            self.act_fun = F.elu
        else:
            raise ValueError(f'Activation function {act_fun} is not known')

    def __init_conv__(self, **params):
        raise NotImplementedError('Must be implemented in the sub class')

    def __extract_conv_results__(self, conv_results):
        return conv_results, None

    def forward(self, data, **other_parms):

        edge_index = data.edge_index
        h_list = []
        e_w_list = []

        transformed_x = F.dropout(data.x, p=self.p_input_dropout, training=self.training)
        transformed_x = self.act_fun(self.input_lin(transformed_x))

        if self.skip_connections:
            other_parms['x_0']=transformed_x

        layer_input = transformed_x
        y_pred_list = []
        for i, l in enumerate(self.layers_list):

            # compute the output
            layer_output, e_w = self.__extract_conv_results__(l(x=layer_input, edge_index=edge_index, **other_parms))
            if e_w is not None:
                e_w_list.append(e_w)

            # manage the output
            layer_output = self.act_fun(layer_output)
            h_list.append(layer_output)

            layer_output = F.dropout(layer_output, p=self.p_dropout, training=self.training)
            layer_input = layer_output

            if self.classifier_at_each_layer:
                y_pred_list.append(self.classifiers_list[i](layer_output))

        if not self.classifier_at_each_layer:
            y_pred_list.append(self.classifiers_list[-1](h_list[-1]))

        return h_list, y_pred_list, e_w_list if len(e_w_list) > 0 else [None]

    @staticmethod
    def get_training_fun():
        return end_to_end_training
