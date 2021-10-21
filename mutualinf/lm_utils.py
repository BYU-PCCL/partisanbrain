def get_device_map(gpus, n_layers):
    '''
    Given a list of gpus, make a dictionary map from GPU to layer, covering all layers evenly.
    params:
        gpus (list): list of gpus
        n_layers (int): number of layers
    return:
        device_map (dict): dictionary mapping from GPU to layer
    '''
    layers_per_gpu = n_layers // len(gpus)
    device_map = {
        gpu: [i for i in range(layers_per_gpu * gpu_num, layers_per_gpu * (gpu_num + 1))] for gpu_num, gpu in enumerate(gpus)
    }
    return device_map

if __name__ == '__main__':
    test_gpus = [0, 1, 2, 3]
    n_layers = 10
    device_map = get_device_map(test_gpus, n_layers)
    print(device_map)