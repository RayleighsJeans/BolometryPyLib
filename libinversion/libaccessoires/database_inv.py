""" **************************************************************************
    header """

import json

""" ********************************************************************** """


def import_database():
    """ database function that returns a dict with info and link on selection
    Args:
      None.
    Returns:
        databas (0, dict): magnetic configuration
    Notes
        None
    """

    database = {'label': 'database',
                'values': {'magnetic_configurations': {}}}

    database['values']['magnetic_configurations']['EIM_000'] = \
        {
        'name': 'EIM_beta000',
        'description': 'beta-scan with bell-shaped pressure profile ~ (1-s)^2; peaking factor = 3; runs with _l6ns151 have an enlarged volume - cover island region\nEIM configuration (ref. configuration A: standard, planned OP1.2 configuration). Flux surface geometry of EIM configuration corresponds to the experimental EJM configuration',
        'URI': '1000_1000_1000_1000_+0000_+0000/01/00',
        'beta': 0.0,
        'w7x_ref': 'w7x_ref_172',
        'peaking_factor': 3.011,
        'pressure_gradient': None,
        'volume': 'increased'
         }

    database['values']['magnetic_configurations']['EIM_044'] = \
        {
        'name': 'EIM_beta044',
        'description': 'beta-scan with bell-shaped pressure profile ~ (1-s)^2; peaking factor = 3; runs with  _l6ns151 have an enlarged volume - cover island region\n EIM configuration (ref. configuration A: standard, planned OP1.2 configuration). Flux surface geometry of EIM configuration corresponds to the experimental EJM configuration',
        'URI': '1000_1000_1000_1000_+0000_+0000/10/06_l6ns151',
        'beta': 0.44,
        'peaking_factor': 4.580,
        'pressure_gradient': None,
        'volume': 'increased'
    }

    database['values']['magnetic_configurations']['KJM_202'] = \
        {
        'name': 'KJM_beta202',
        'description': 'another pressure profile guess; n/ne0=(1-s^4); Ti/Ti0=(1-s^4)^2; Te/Te0= 0.8*(1-s)^10 + 0.2*(1-s^4)^2; p=nk(Te+Ti); parameters n0=1e+1 Te0=8keV, Ti0=1keV leads to 14.4kPa. Further calculations by scaling this profile, which would mean e.g. different densities\n Profile-case for EUTERPE runs. Maximum pressure gradient at s=0.5 . Peaking factor is 2\n KJM configuration (ref. configuration E: high-mirror). Flux surface geometry of KJM configuration corresponds to the experimental KKM configuration',
        'URI': '0972_0926_0880_0852_+0000_+0000/05/0954ams',
        'beta': 2.02,
        'w7x_ref': 'w7x_ref_115',
        'peaking_factor': 2.044,
        'pressure_gradient': 0.5,
        'volume': 'standard'
    }

    database['values']['magnetic_configurations']['FTM_060'] = \
        {
        'name': 'FTM_beta060',
        'description': 'beta-scan with bell-shaped pressure profile ~ (1-s)^2; peaking factor = 3; runs ending with l8ns148 have an enlarged volume - cover island region\nTM configuration (ref. configuration C: high-iota, planned OP1.2 configuration)',
        'URI': '1000_1000_1000_1000_-0690_-0690/02/06l8ns148',
        'beta': 0.60,
        'w7x_ref': 'w7x_ref_417',
        'peaking_factor': 4.605,
        'pressure_gradient': None,
        'volume': 'increased'
    }

    database['values']['magnetic_configurations']['AIM_207'] = \
        {
        'name': 'AIM_beta207',
        'description': 'beta-sequence with pressure profile form 1-s AIM configuration (ref. configuration D: low-mirror)',
        'URI': '1042_1042_1127_1127_+0000_+0000/01/10',
        'beta': 2.07,
        'w7x_ref': 'w7x_ref_23',
        'peaking_factor': 2.057,
        'pressure_gradient': None,
        'volume': 'standard'
    }

    database['values']['magnetic_configurations']['EIM_022'] = \
        {
        'name': 'EIM_beta022',
        'description': 'beta-scan with bell-shaped pressure profile ~ (1-s)^2; peaking factor = 3; runs ending with l8ns148 have an enlarged volume - cover island region\nEIM configuration (ref. configuration A: standard, planned OP1.2 configuration). Flux surface geometry of EIM configuration corresponds to the experimental EJM configuration',
        'URI': '1000_1000_1000_1000_+0000_+0000/10/03_l6ns151',
        'beta': 0.22,
        'w7x_ref': 'w7x_ref_174',
        'peaking_factor': 4.548,
        'pressure_gradient': None,
        'volume': 'increased'
    }

    database['values']['magnetic_configurations']["KJM_000"] = \
        {
        "name": "KJM_beta000",
        "description": "KJM configuration (ref configuration E: high-mirror, OP1.2 configuration) Flux surface geometry of KJM config. corresponds to the experimental KKM configuration.",
        "URI": "0972_0926_0880_0852_+0000_+0000/01/00jh",
        "beta": 0.00,
        'w7x_ref': 'w7x_ref_27',
        "peaking_factor": 2.007,
        "minor_radius": 0.49,
        "major_radius": 5.49,
        "coil_currents": [
            4400,
            14000,
            13333.33,
            12666.67,
            12266.67,
            0,
            0
        ],
        "volume_LCFS": 26.10,
        "pressure_gradient": 0.5,
        "volume": "standard"
    }

    database['values']['magnetic_configurations']["KJM_027"] = \
        {
        "name": "KJM_beta027",
        "description": "KJM configuration (ref. configuration E: high-mirror, OP1.2 configuration). Flux surface geometry of KJM configuration corresponds to the experimental KKM configurationn.",
        "URI": "0972_0926_0880_0852_+0000_+0000/02n/02",
        "beta": 0.27,
        'w7x_ref': 'w7x_ref_336',
        "peaking_factor": 3.024,
        "minor_radius": 0.50,
        "major_radius": 5.49,
        "coil_currents": [
            14400,
            14000,
            13333.33,
            12666.67,
            12266.67,
            0,
            0
        ],
        "volume_LCFS": 26.81,
        "pressure_gradient": 0.5,
        'B0': -2.41,
        "volume": "standard"
    }

    database['values']['magnetic_configurations']["EIM_065"] = \
        {
        "name": "EIM_beta065",
        "description": "EIM configuration (ref. configuration A: standard, OP1.2 configuration). Flux surface geometry of EIM configuration corresponds to the experimental EJM configuration.",
        "URI": "1000_1000_1000_1000_+0000_+0000/01/04m",
        "beta": 0.65,
        'w7x_ref': 'w7x_ref_7',
        "peaking_factor": 2.0024,
        "minor_radius": 0.54,
        "major_radius": 5.52,
        "coil_currents": [
            15000,
            15000,
            15000,
            15000,
            15000,
            0,
            0
        ],
        "B0": -2.7,
        "volume_LCFS": 31.86,
        "pressure_gradient": 0.5,
        "volume": "standard"
    }

    database['values']['magnetic_configurations']["DBM_000"] = \
        {
        "name": "DBM_beta000",
        "description": "DBM configuration (ref. configuration B: low-iota, OP1.2 configuration)",
        "URI": "1000_1000_1000_1000_+0750_+0750/01/00",
        "beta": 0.00,
        'w7x_ref': 'w7x_ref_18',
        "peaking_factor": 2.005,
        "minor_radius": 0.54,
        "major_radius": 5.5,
        "coil_currents": [
            -12222.22,
            -12222.22,
            -12222.22,
            -12222.22,
            -12222.22,
            -9166.67,
            -9166.67
        ],
        "B0": -2.45,
        "volume_LCFS": 31.91,
        "volume": "standard"
    }

    with open('database.json', 'w') as parsedfile:
        json.dump(database, parsedfile, indent=4, sort_keys=False)

    return database
