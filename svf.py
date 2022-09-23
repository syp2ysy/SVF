import copy
import inspect
from math import floor

import paddle
import paddle.nn as nn


def d_nsvd(matrix, rank=1):
    U, S, V = paddle.linalg.svd(matrix)
    S = S[:rank]
    U = U[:, :rank]  # * S.view(1, -1)
    V = V[:, :rank]  # * S.view(1, -1)
    V = paddle.transpose(V, perm=[1, 0])
    return U, S, V


class SVD_Conv2d(nn.Lyaer):
    """Kernel Number first SVD Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias,
                 padding_mode='zeros', device=None, dtype=None,
                 rank=1):
        super(SVD_Conv2d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.conv_U = nn.Conv2d(rank, out_channels, (1, 1), (1, 1), 0, (1, 1), 1, bias)
        self.conv_V = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, dilation, groups, False)
        self.vector_S = nn.ParameterList(paddle.empty((1, rank, 1, 1), **factory_kwargs))

    def forward(self, x):
        x = self.conv_V(x)
        x = x.mul(self.vector_S)
        output = self.conv_U(x)
        return output

class SVD_Linear(nn.Layer):

    def __init__(self, in_features, out_features, bias, device=None, dtype=None, rank=1):
        super(SVD_Linear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.fc_V = nn.Linear(in_features, rank, False)
        self.vector_S = nn.ParameterList(paddle.empty((1, rank), **factory_kwargs))
        self.fc_U = nn.Linear(rank, out_features, bias)

    def forward(self, x):
        x = self.fc_V(x)
        x = x.mul(self.vector_S)
        output = self.fc_U(x)
        return output


full2low_mapping_n = {
    nn.Conv2d: SVD_Conv2d,
    nn.Linear: SVD_Linear
}


def replace_fullrank_with_lowrank(model, full2low_mapping={}, layer_rank={}, lowrank_param_dict={},
                                  module_name=""):
    """Recursively replace original full-rank ops with low-rank ops.
    """
    if len(full2low_mapping) == 0 or full2low_mapping is None:
        return model
    else:
        for sub_module_name in model._modules:
            current_module_name = sub_module_name if module_name == "" else \
                module_name + "." + sub_module_name
            # has children
            if len(model._modules[sub_module_name]._modules) > 0:
                replace_fullrank_with_lowrank(model._modules[sub_module_name],
                                              full2low_mapping,
                                              layer_rank,
                                              lowrank_param_dict,
                                              current_module_name)
            else:
                if type(getattr(model, sub_module_name)) in full2low_mapping and \
                        current_module_name in layer_rank.keys():
                    _attr_dict = getattr(model, sub_module_name).__dict__
                    # use inspect.signature to know args and kwargs of __init__
                    _sig = inspect.signature(
                        type(getattr(model, sub_module_name)))
                    _kwargs = {}
                    for param in _sig.parameters.values():
                        if param.name not in _attr_dict.keys():
                            if 'bias' in param.name:
                                if getattr(model, sub_module_name).bias is not None:
                                    value = True
                                else:
                                    value = False
                            elif 'stride' in param.name:
                                value = 1
                            elif 'padding' in param.name:
                                value = 0
                            elif 'dilation' in param.name:
                                value = 1
                            elif 'groups' in param.name:
                                value = 1
                            elif 'padding_mode' in param.name:
                                value = 'zeros'
                            else:
                                value = None
                            _kwargs[param.name] = value
                        else:
                            _kwargs[param.name] = _attr_dict[param.name]
                    _kwargs['rank'] = layer_rank[current_module_name]
                    _layer_new = full2low_mapping[type(
                        getattr(model, sub_module_name))](**_kwargs)
                    old_module = getattr(model, sub_module_name)
                    old_type = type(old_module)
                    bias_tensor = None
                    if _kwargs['bias'] == True:
                        bias_tensor = old_module.bias.data
                    setattr(model, sub_module_name, _layer_new)
                    new_module = model._modules[sub_module_name]
                    if old_type == nn.Conv2d:
                        conv1 = new_module._modules["conv_V"]
                        conv2 = new_module._modules["conv_U"]
                        param_list = lowrank_param_dict[current_module_name]
                        conv1.weight.data.copy_(param_list[1])
                        conv2.weight.data.copy_(param_list[0])
                        new_module.vector_S.data.copy_(param_list[2])
                        if bias_tensor is not None:
                            conv2.bias.data.copy_(bias_tensor)
    return model


class DatafreeSVD(object):

    def __init__(self, model, global_rank_ratio=1.0,
                 excluded_layers=[], customized_layer_rank_ratio={}, skip_1x1=True, skip_3x3=True):
        # class-independent initialization
        super(DatafreeSVD, self).__init__()
        self.model = model
        self.layer_rank = {}
        model_dict_key = list(model.state_dict().keys())[0]
        model_data_parallel = True if str(
            model_dict_key).startswith('module') else False
        self.model_cpu = self.model.module.to(
            "cpu") if model_data_parallel else self.model.to("cpu")
        self.model_named_modules = self.model_cpu.named_modules()
        self.rank_base = 4
        self.global_rank_ratio = global_rank_ratio
        self.excluded_layers = excluded_layers
        self.customized_layer_rank_ratio = customized_layer_rank_ratio
        self.skip_1x1 = skip_1x1
        self.skip_3x3 = skip_3x3

        

        self.param_lowrank_decomp_dict = {}
        registered_param_op = [nn.Conv2d, nn.Linear]

        for m_name, m in self.model_named_modules:
            if type(m) in registered_param_op and m_name not in self.excluded_layers:
                weights_tensor = m.weight.data
                tensor_shape = weights_tensor.squeeze().shape
                param_1x1 = False
                param_3x3 = False
                depthwise_conv = False
                if len(tensor_shape) == 2:
                    full_rank = min(tensor_shape[0], tensor_shape[1])
                    param_1x1 = True
                elif len(tensor_shape) == 4:
                    full_rank = min(
                        tensor_shape[0], tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
                    if tensor_shape[2] == 1 and tensor_shape[3] == 1:
                        param_1x1 = True
                    else:
                        param_3x3 = True
                else:
                    full_rank = 1
                    depthwise_conv = True

                if self.skip_1x1 and param_1x1:
                    continue
                if self.skip_3x3 and param_3x3:
                    continue
                if depthwise_conv:
                    continue

                low_rank = round_to_nearest(full_rank,
                                                ratio=self.global_rank_ratio,
                                                base_number=self.rank_base,
                                                allow_rank_eq1=True)

                self.layer_rank[m_name] = low_rank

    def decompose_layers(self):
        self.model_named_modules = self.model_cpu.named_modules()
        for m_name, m in self.model_named_modules:
            if m_name in self.layer_rank.keys():
                weights_tensor = m.weight.data
                tensor_shape = weights_tensor.shape
                if len(tensor_shape) == 1:
                    self.layer_rank[m_name] = 1
                    continue
                elif len(tensor_shape) == 2:
                    weights_matrix = m.weight.data
                    U, S, V = d_nsvd(weights_matrix, self.layer_rank[m_name])
                    self.param_lowrank_decomp_dict[m_name] = [
                        U, V, S.reshape(1, self.layer_rank[m_name])]
                elif len(tensor_shape) == 4:
                    weights_matrix = m.weight.data.reshape(tensor_shape[0], -1)
                    U, S, V = d_nsvd(weights_matrix, self.layer_rank[m_name])
                    self.param_lowrank_decomp_dict[m_name] = [
                        U.reshape(tensor_shape[0],
                                  self.layer_rank[m_name], 1, 1),
                        V.reshape(
                            self.layer_rank[m_name], tensor_shape[1], tensor_shape[2], tensor_shape[3]),
                        S.reshape(1, self.layer_rank[m_name], 1, 1)    
                    ]

    def reconstruct_lowrank_network(self):
        self.low_rank_model_cpu = copy.deepcopy(self.model_cpu)
        self.low_rank_model_cpu = replace_fullrank_with_lowrank(
            self.low_rank_model_cpu,
            full2low_mapping=full2low_mapping_n,
            layer_rank=self.layer_rank,
            lowrank_param_dict=self.param_lowrank_decomp_dict,
            module_name=""
        )
        return self.low_rank_model_cpu

def round_to_nearest(n, ratio=1.0, base_number=4, allow_rank_eq1=False):
    rank = floor(floor(n * ratio) / base_number) * base_number
    rank = min(max(rank, 1), n)
    if rank == 1:
        rank = rank if allow_rank_eq1 else n
    return rank

def resolver(
        model,
        global_low_rank_ratio=1.0,
        excluded_layers=[],
        customized_layers_low_rank_ratio={},
        skip_1x1=False,
        skip_3x3=False
):
    lowrank_resolver = DatafreeSVD(model,
                                   global_rank_ratio=global_low_rank_ratio,
                                   excluded_layers=excluded_layers,
                                   customized_layer_rank_ratio=customized_layers_low_rank_ratio,
                                   skip_1x1=skip_1x1,
                                   skip_3x3=skip_3x3)
    lowrank_resolver.decompose_layers()
    lowrank_cpu_model = lowrank_resolver.reconstruct_lowrank_network()
    return lowrank_cpu_model


if __name__ == "__main__":
    origin_model = FSS_model
    final_model = resolver(origin_model)
