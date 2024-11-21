import torch

def get_constraints(model, input_domain):
    abstract_element = input_domain
    print(abstract_element)
    with torch.no_grad():
        for layer in model.layers:
            abstract_element = layer(abstract_element)
            print(layer)
            print(abstract_element)
    return abstract_element
