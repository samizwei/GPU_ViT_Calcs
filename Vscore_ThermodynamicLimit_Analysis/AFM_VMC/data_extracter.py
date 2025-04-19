import os
import json
import numpy as np

def extract_data(file_list, data_dict, folder_path, list_to_replace, replacers, L=0):
    for file in file_list:
        with open(os.path.join(folder_path, file), 'r') as f:
            data = json.load(f)
            short_name = file[:file.index('.log')]
            for j,item in enumerate(list_to_replace):
                short_name = short_name.replace(item, replacers[j])
                # short_name = short_name.replace(item, '')
            
            data_dict[short_name] = {
                'Energy': data['Energy']['Mean']['real'],
                'Variance': data['Energy']['Variance'],
                'iters': data['Energy']['iters'],
                'acceptance': data['acceptance']['value']
            }
            vscores = Vscore(var = np.array(data['Energy']['Variance']),
                             E = np.array(data['Energy']['Mean']['real']),
                             n_dof = L**2, Einf = 1e-3)
            
            data_dict[short_name]['Vscore'] = vscores


def Vscore(var, E, n_dof, Einf=0.0):
    """
    intputs:
    var: (array) variance of the Hamiltonian for each iteration step
    E: (array) energy of the Hamiltonian for each iteration step
    n_dof: (int) number of degrees of freedom, usually the number of sites in the lattice
    Einf: (float) energy of the infinite system, if None set to Zero

    returns:
    Vscore: (array) Vscore for each iteration step
    """
    if len(var) != len(E):
        raise ValueError("var and E must be of the same length")


    return n_dof * var / (E - Einf)**2