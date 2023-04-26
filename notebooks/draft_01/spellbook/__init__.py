from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


import importlib
from copy import deepcopy

def instantiate(class_dict, **kwargs):
    # Extract class path, module, and class name
    class_dict = deepcopy(class_dict)
    assert '_target_' in class_dict, "Man, you need a _target_ attr to specify which class you want to instantiate"
    class_path = class_dict.pop('_target_')
    module_name, class_name = class_path.rsplit('.', 1)

    # Import the module and class dynamically
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    # Instantiate the object with the remaining kwargs
    instance = class_(**class_dict, **kwargs)

    return instance