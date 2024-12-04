from abstract_interpretation import domains

obs_space_domain = domains.DeepPoly([0, 0, 0],[5, 5, 5])
safe_space_domain = domains.DeepPoly([0, 0, 0],[1, 5, 5])

print(domains.recover_safe_region(obs_space_domain, [safe_space_domain]))

unsafe_domains = domains.recover_safe_region(obs_space_domain, [safe_space_domain])

print(domains.recover_safe_region(obs_space_domain, unsafe_domains))
