import torch.nn.functional as F
import torch
from captum.attr import LayerActivation, LayerGradientXActivation


def getAllLayers(net):
    layers = []
    for layer in net.children():
        if len(list(layer.children())) == 0 and (isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear)):
            layers.append(layer)
        if len(list(layer.children())) > 0:
            temp_layers = getAllLayers(layer)
            layers = layers + temp_layers
    return layers


global Hooks_Storage
Hooks_Storage = []


def getInputAndOutputofLayer(self, input, output):
    global Hooks_Storage
    assert len(
        input) == 1, f"Hook, {self.__class__.__name__} Expected 1 input, got {len(input)}"
    Hooks_Storage.append(output.detach())


def insertHooks(layers):
    all_hooks = []
    for layer in layers:
        h = layer.register_forward_hook(getInputAndOutputofLayer)
        all_hooks.append(h)
    return all_hooks


def Scale(out, rmax=1, rmin=0):
    output_std = (out - out.min()) / (out.max() - out.min())
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled


def my_eval_NeuronsActivations(model, x):
    global Hooks_Storage
    layer2output = []
    all_layers = getAllLayers(model)

    layers = all_layers  # [1:]

    hooks = insertHooks(layers)
    model(x)  # forward pass and everthing is stored in Hooks_Storage
    for l_id in range(len(layers)):
        outputs = F.relu(Hooks_Storage[l_id])
        # outputs = torch.squeeze(torch.sum(outputs, dim=1)) # present in orignal implementation
        scaled_outputs = Scale(outputs)

        layer2output.append(scaled_outputs)

    _ = [h.remove() for h in hooks]  # remove the hooks
    Hooks_Storage = []
    return torch.cat([out.flatten() for out in layer2output]), layer2output


def my_eval_2_CaptumNeuronActivatioin(model, img):
    img.requires_grad = True
    img.grad = None

    layer2output = []
    all_layers = getAllLayers(model)
    for lid, layer in enumerate(all_layers):
        cond = LayerGradientXActivation(model, layer)
        cond_vals = cond.attribute(img, target=1)

        layer2output.append(cond_vals.flatten())

    return torch.cat([out for out in layer2output]), layer2output


def getNeuronCoverage(model, img, device):

    r = my_eval_2_CaptumNeuronActivatioin(model, img)

    torch.cuda.synchronize()

    return r


