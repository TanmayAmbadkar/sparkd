import torch
from abstract_interpretation import domains

def get_constraints(model, input_domain):
    abstract_element = input_domain
    # print(abstract_element)
    with torch.no_grad():
        for layer in model.layers:
            abstract_element = layer(abstract_element)
    return abstract_element

def get_ae_bounds(model, input_domain):
    
    input_domain.lower = (input_domain.lower - model.mean)/(model.std)
    input_domain.upper = (input_domain.lower - model.mean)/(model.std)
    
    domain = get_constraints(model.encoder.shared_net, input_domain)
    mu_domain = get_constraints(model.encoder.fc_mu, domain)
    
    return mu_domain.calculate_bounds()
    

def get_variational_bounds(model, input_domain):
    domain = get_constraints(model.encoder.shared_net, input_domain)
    mu_domain = get_constraints(model.encoder.fc_mu, domain)
    # logvar_domain = get_constraints(model.encoder.fc_logvar, domain)
    
    mu_domain = domains.DeepPoly(*mu_domain.calculate_bounds())
    # logvar_domain = domains.DeepPoly(*logvar_domain.calculate_bounds())
    # sigma_bounds = domains.DeepPoly(torch.exp(0.5 * logvar_domain.lower), torch.exp(0.5 * logvar_domain.upper))
    mu_min, mu_max = mu_domain.lower, mu_domain.upper
    # sigma_min, sigma_max = sigma_bounds.lower, sigma_bounds.upper
    epsilon_min, epsilon_max = -1.96, 1.96

    # z_min = mu_min + sigma_min * epsilon_min
    # z_max = mu_max + sigma_max * epsilon_max
    
    return mu_min, mu_max
    
    # return z_min, z_max
