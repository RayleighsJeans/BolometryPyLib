""" ********************************************************************** """

import sys
import numpy as np
import requests

import scipy.io.netcdf as netCDF
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline as RBS

import periodictable as period
import pandas as pd
from glob import glob as glob

Z = np.zeros
M = 10000
stdwrite = sys.stdout.write

# strahl_loc = 'C:/Users/pih/ownCloud/strahl/'
# strahl_loc = '//sv-di-fs-1/E5/E5-IMP/Bolometer/' + \
#     'Besprechungen/SubtopicalGroupMeeting/Dec2019/W2/strahl/'
# strahl_loc = '//sv-di-fs-1/E5/E5-IMP/Bolometer/' + \
#     'Besprechungen/SubtopicalGroupMeeting/Jan2020/W2/strahl/'
# strahl_loc = '//sv-di-fs-1/E5/E5-IMP/Bolometer/' + \
#     'Besprechungen/SubtopicalGroupMeeting/Jan2020/W3/strahl/'
# strahl_loc = '//sv-di-fs-1/E5/E5-IMP/Bolometer/' + \
#     'Besprechungen/SubtopicalGroupMeeting/Jan2020/W4/strahl/'
# strahl_loc = '//sv-di-fs-1/E5/E5-IMP/Bolometer/' + \
#     'Besprechungen/SubtopicalGroupMeeting/Feb2020/W1/strahl/'
strahl_loc = '//sv-di-fs-1/E5/E5-IMP/Diagnostics/Bolometer/Software/strahl/'

""" ********************************************************************** """


def read_param_file(
        base='param_files/',
        file='strahl_run_00009.0',
        debug=False):
    """ Reading parameter file
    Keyword Arguments:
        base {str} -- location base (default: {'param_files/'})
        file {str} -- file descriptor (default: {'strahl_run_00009.0'})
        debug {bool} -- debugging bool (default: {False})
    Returns:
        read_out {list} -- list of line strings from file
    """
    loc = strahl_loc + base + file
    with open(loc, 'r') as param_file:
        read_in = param_file.readlines()
    param_file.close()

    read_out = []
    for i, line in enumerate(read_in):
        L = line.replace('\n', '').replace(
            '\n', '').replace('\n', '')
        if debug:
            print(line, '\n', L, '\n\n')
        if L != '':
            read_out.append(L)

    if debug:
        print(read_out)

    return (np.array(read_out))


def sort_param_file(
        base='param_files/',
        file='strahl_run_00009.0',
        attribute='D    D_SOL',
        time=None,  # '# of changes',
        grid=None,  # 'rho',
        dtype='string',
        debug=False):
    """ Sorting through the param file and try to find the attribute
    Keyword Arguments:
        base {str} -- directory base (default: {'param_files'})
        file {str} -- run file (default: {'strahl_run_00009.0'})
        attribute {str} -- line header/content to look for (default: {'D    D_SOL'})
        time {str} -- time line start to look for (default: {'# of changes'})
        grid {str} -- grid line start to look for (default: {'rho'})
        debug {bool} -- debuggin bool (default: {False})
    Returns:
        result {ndarray} -- 1D/2D array of returned quantity (default: ndarray(x(, y)))
        read_in {list} -- list of read in param file (default: list(z))
        headers {list} -- list of cv file headers (default: list(w))
        cv_idx {list} --  cv lines index in read in (default: list(w))
        primary {int} -- index where attribute is found (default: int)
        secondary {int} -- index where grid porperty is found (default: int)
        number {int} -- number of time vector points (default: int)
    """
    read_in = read_param_file(
        base=base, file=file, debug=False)

    cv_idx = [i for i, L in enumerate(read_in) if 'cv' in L]
    headers = read_in[cv_idx]

    if debug:
        for i, L in enumerate(headers):
            print(cv_idx[i], L)

    # where u search for is located
    primary = [cv_idx[i] for i, L in enumerate(headers)
               if attribute in L][0]
    if debug:
        print('primary =', primary, '\nread_in[primary + 1] =',
              read_in[primary + 1])

    if dtype == 'str':
        return (read_in[primary + 1].split()[0].replace("'", ""),
                read_in, headers, cv_idx, primary)

    if time is not None:
        secondary = [cv_idx[i] for i, L in enumerate(
                     headers[:cv_idx.index(primary) - 1])
                     if time in L][-1]
        if debug:
            print('time list=', headers[:cv_idx.index(primary) - 1],
                  '\nsecondary=', secondary)

        number = int(read_in[secondary + 1].split()[0])
        data = read_in[primary + 1: primary + number + 1][0].split()
        if debug:
            print('number=', number,
                  '\nresult=', data)

    if grid is not None:
        if (grid in headers[cv_idx.index(primary) - 1]):
            grid = read_in[cv_idx[cv_idx.index(primary) - 1] + 1].split()

            result = [np.array([float(v) for v in data]),
                      np.array([float(v) for v in grid])]

        else:
            result = np.array([float(v) for v in data])

        return (result, read_in, headers, cv_idx,
                primary, secondary, number)

    else:
        return (np.array([float(v) for v in data]),
                read_in, headers, cv_idx, primary)


def read_result_file(
        file=None,  # 'C_00005t2.421_3.421_1',
        material=None,  # 'C_',
        strahl_id=None,  # '00005',
        base='result/',
        debug=False):
    """ Reading netCDF result file
    Keyword Arguments:
        file {str} -- file descriptor (default: {'C_00005t2.421_3.421_1'})
        base {str} -- location base (default: {'result/'})
        debug {bool} -- debugging bool (default: {False})
    Returns:
        results {netCDF.object} -- netCDF object
        run_id {str} -- run ID
        material {str} -- element
    """
    if (material is None) or (id is None):
        print('\t>>Loading ' + file)
        loc = strahl_loc + base + file
        run_id = file[2:]
    elif isinstance(material, str) and isinstance(strahl_id, str):
        loc = glob(strahl_loc + base + material + strahl_id + 't*')[0]
        file = run_id = loc[-19:]
        print('\t>>Loading ' + run_id)
    else:
        print('\\\ failed looking for file, error in mat/id/loc')
        return (None)

    results = netCDF.netcdf_file(loc, 'r', mmap=False)
    for i, tag in enumerate(results.__dict__.keys()):
        if debug:
            if (tag == 'variables'):
                for j, var_tag in enumerate(results.variables.keys()):
                    if (j == 0):
                        print('\n\t=== VARIABLES ===')
                    print('\t', var_tag, '\t', np.shape(
                        results.variables[var_tag]))
            elif (tag == 'dimensions'):
                for j, dim_tag in enumerate(results.dimensions.keys()):
                    if (j == 0):
                        print('\n\t=== DIMENSIONS ===')
                    print('\t', dim_tag, '\t', results.dimensions[dim_tag])
            else:
                print(tag)
    return (results, file.replace('C_', ''), file[0:2].replace('_', ''),
            run_id)


def return_attribute(
        netcdf=None,
        var_name='electron_density',
        debug=False):
    """ Reads attribute/quantity from netCDF object
    Keyword Arguments:
        netcdf {netCDF object} --  (default: {None})
        var_name {str} -- variable name to look for (default: {'electron_density'})
        debug {bool} -- debugging bool (default: {False})
    Returns:
        attribue {ndarray}/{float}/{list} -- returned attribute (default: np.array((x, y))
        shape {tuple}/{npfloat} -- shape of the attribute (default: (x, y)
    """
    if debug:
        print('__dict__.keys():')
        for i, key in enumerate(netcdf.__dict__.keys()):
            print('\t', key)

        print('variables.keys():')
        for i, key in enumerate(netcdf.variables.keys()):
            print('\t', key)

        print('dimensions.keys():')
        for i, key in enumerate(netcdf.dimensions.keys()):
            print('\t', key)

    if (var_name in netcdf.__dict__.keys()):
        if debug:
            print('__dict__:', var_name, ',shape:', np.shape(
                  netcdf.__dict__[var_name]))
        return (netcdf.__dict__[var_name], None, None, None)

    elif (var_name in netcdf.variables.keys()):
        if len(np.shape(netcdf.variables[var_name])) == 3:
            n_imps = np.shape(netcdf.variables[var_name][:])[1]

            if debug:
                print(var_name, ',n imps:', n_imps, ',shape:',
                      np.shape(netcdf.variables[var_name][:]),
                      ',new shape:', np.shape(
                          netcdf.variables[var_name][-1]))
            return (netcdf.variables[var_name][-1].transpose(),
                    netcdf.dimensions['grid_points'],
                    len(netcdf.variables['time'][:]),
                    np.shape(netcdf.variables[var_name]))

        elif len(np.shape(netcdf.variables[var_name])) == 2:
            if debug:
                print('variables:', var_name, ',shape:', np.shape(
                      netcdf.variables[var_name][:]))
            return (netcdf.variables[var_name][:][-1],
                    np.shape(netcdf.variables[var_name][:][-1]))

        elif len(np.shape(netcdf.variables[var_name])) == 1:
            if debug:
                print('variables:', var_name, ',shape:', np.shape(
                      netcdf.variables[var_name][:]))
            return (netcdf.variables[var_name][:],
                    np.shape(netcdf.variables[var_name][:]))

    elif (var_name in netcdf.dimensions.keys()):
        if debug:
            print('dimensions:', var_name, ',shape:', np.shape(
                  netcdf.dimensions[var_name]))
        return (netcdf.dimensions[var_name], None, None, None)

    else:
        print("\t\\\ couldn't find " + var_name)
        return (None, None)


def corona_core_v_sol(
        grid_data=Z((300)),
        data=Z((12, 300)),
        r_maj=520.69,
        vmecID='1000_1000_1000_1000_+0000_+0000/01/00jh_l/',
        debug=False):
    """ STRAHL core v SOL radiation ratios and torus volume
    Keyword Arguments:
        grid_data {[type]} -- radial grids (default: {Z((300))})
        data {[type]} -- data of radiation (default: {Z((12, 300))})
        r_maj {float} -- minor radius (default: {520.69})
        vmecID {str} -- (def.: {'1000_1000_1000_1000_+0000_+0000/01/00jh_l/'})
        debug {bool} -- debugging (default: {False})
    Returns:
        power {ndarray} -- 0: core power, 1: SOL, 2, core ratio, 3: SOL
        volume {ndarray} -- grid volume
    """
    try:
        magax = requests.get(
            'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/' +
            vmecID + 'magneticaxis.json?phi=108').json()['magneticAxis']
    except Exception:
        print('\t\\\ failed magax')
        magax = {  # EIM beta = 0.0
            'x1': 5.204812678833707, 'x3': 9.978857072810671e-17}

    def circ_torus_vol(
            r_mn=1.0,
            r_mj=5.0):
        return (2 * np.pi**2 * r_mn**2 * r_mj)

    volume = Z((np.shape(grid_data)[0]))
    power = Z((4, np.shape(data)[1]))

    shells = grid_data[:] * r_maj / 1000.
    for i, r in enumerate(shells[1:]):
        volume[i] = circ_torus_vol(r_mn=r, r_mj=magax['x1'][0]) - \
            circ_torus_vol(r_mn=shells[i], r_mj=magax['x1'][0])

        for j, imp in enumerate(np.transpose(data[:, :])):
            if r <= r_maj / 1000.:
                power[0, j] += volume[i] * imp[i]
            else:
                power[1, j] += volume[i] * imp[i]

    for i in range(np.shape(data)[1]):
        power[2, i] = power[0, i] / (power[0, i] + power[1, i])
        power[3, i] = power[1, i] / (power[0, i] + power[1, i])

    # print(np.sum(power[0:2, :], axis=1))

    return (power, volume)


def scale_impurity_radiation(
        file=None,  # 'C_00005t2.421_3.421_1',
        material=None,  # 'C_'
        strahl_id=None,  # '00005'
        debug=False):
    """ Loads, calculates and scales the (total)
        impurity radiation from STRAHL
    Keyword Arguments:
        file {str} -- file to load from (default: {'C_00005t2.421_3.421_1'})
        debug {bool} -- debugging bool (default: {False})
    Returns:
        data {ndarray} -- array of each impurity for the
                          poloidal R (default: ndarray(x, y))
        grid_data {list} -- poloidal grid (default: ndarray(y))
        labels {list} -- list of impurity labels
                         and L-int values (default: list(x))
        r_maj {np.float} -- major radius (default: np.float)
        el_symb {list} -- list of element symbols (default: list(x))
        mass {list} -- list of masses for elements/ions (default: list(x))
        density {list} -- list of mass/volume densities (default: list(x))
        number_density {list} -- list of number/volume densities
                                 (default: list(x))
        results {netCDF object} -- netCDF object (default: netCDF object)
    """
    if (material is None) or (id is None):
        data, grid_data, imp_rad, results, run_id = \
            impurity_radiation_corona(file=file, debug=False)
    elif isinstance(material, str) and isinstance(strahl_id, str):
        data, grid_data, imp_rad, results, run_id = \
            impurity_radiation_corona(
                material=material, strahl_id=strahl_id, debug=False)
    else:
        print('\\\ failed looking for file, error in mat/id/loc')
        return (None)

    zmax = return_attribute(netcdf=results, var_name='maximum_charge')[0]
    r_maj = return_attribute(netcdf=results, var_name='large_radius')[0]

    # label = strarr(ny_tot)
    charge = np.linspace(0, zmax, zmax + 1)
    el_symb = [None] * (zmax + 1)
    element = str(return_attribute(
        netcdf=results, var_name='part_of_filename')[0],
        encoding='utf-8')[0:2]
    el_symb[0] = element
    el_symb[zmax] = ' '

    if zmax >= 1.:
        electrons = zmax - charge[1:zmax]
        # period_system, electrons, ff, /charge
        symbols, _, mass, density, number_density = \
            period_system(charge=electrons, debug=False)

        el_symb[1:zmax - 1] = symbols

    n_data, ny_tot = np.shape(data)[:]
    nlin = return_attribute(netcdf=results, var_name='maximum_charge')[0]
    labels = [None] * ny_tot

    if False:  # keyword_set(rho) then begin ; time plot
        percentages = Z((ny_tot))
        percentages[0:ny_tot - 1] = \
            data[n_data - 1, 0:ny_tot - 1] / data[n_data - 1, nlin + 3] * 100.

        for i in range(nlin):
            labels[i] = " Z= " + int(str(charge[i])) + "(" + \
                str(el_symb[i]) + ") line ->" + \
                format(percentages[i], '.2e') + " %"

    lin_int = strahl_integral(y=data, netcdf=results, lin=True)
    vol_int = strahl_integral(y=data, netcdf=results, vol=True)

    preface = ['C$^{6+}$', 'cont. imp.', 'brems. imp.', 'total rad.',
               'total imp.', 'total imp. (corona)']

    S = el_symb[0].replace('_', '')  # species
    for i in range(nlin + 6):
        if (i == 0):
            labels[i] = S + '  '
        elif (0 < i < nlin):
            labels[i] = S + '$^{' + str(int(charge[i])) + '+}$'
        elif (i >= nlin):
            labels[i] = preface[i - nlin]
        labels[i] += " (" + format(lin_int[i], '.2e') + "W/m$^{2}$; " + \
            format(vol_int[i], '.2e') + "W/m$^{3}$)"

    # scaling
    data *= 1.e6  # in W?
    return (data, grid_data, labels, r_maj,
            el_symb, mass, density, number_density, results, run_id)


def impurity_radiation_corona(
        file=None,  # 'C_00005t2.421_3.421_1',
        material=None,  # 'C_'
        strahl_id=None,  # '00005'
        debug=False):
    """ setting up and calculating the total radiation of impurities
    Keyword Arguments:
        file {str} -- file string if no material/strahl_id (default: {None})
        material {str} -- material/impurity string (default: {None})
        strahl_id {str} -- strahl id (run number) (default: {None})
        debug {bool} -- (default: {False})
    Returns:
        data {ndarray} -- full data array of impurity, imp., brems-... rad.
            data[, :n_imp+3] -- impurity/ion stage diagnostic line radiation
            data[, n_imp+4] -- brems. strahlung
            data[, n_imp+5] = total corona equilibrium radiation
        grid_data {ndarray} -- radius grid for data given
        imp_rad {ndarray} -- diagnostic line radiation of ions from STRAHL
        results {netcdf object} -- loaded netcdf object of file
        run_id {str} -- combined run id from whatever you put inm
    """
    if (material is None) or (id is None):
        results, run_id, mat, run_id = read_result_file(file=file)
    elif isinstance(material, str) and isinstance(strahl_id, str):
        results, run_id, mat, run_id = read_result_file(
            material=material, strahl_id=strahl_id)
    else:
        print('\\\ failed looking for file, error in mat/id/loc')
        return (None)

    grid_data = return_attribute(
        netcdf=results, var_name='rho_poloidal_grid')[0]

    nlin = return_attribute(netcdf=results, var_name='maximum_charge')[0]
    ny_tot, n_data = nlin + 6, np.shape(grid_data)[0]
    imp_rad = return_attribute(
        netcdf=results, var_name='impurity_radiation',
        debug=False)[0][-n_data:, :]

    if debug:
        print('imp_rad=', np.shape(imp_rad),  # , imp_rad
              '\nny_tot=', ny_tot, ' n_data=', n_data, ' nlin=', nlin)

    recomb = str(return_attribute(
        netcdf=results, var_name='recombination_data_file')[0],
        encoding='utf-8').replace(' ', '')
    ion = str(return_attribute(
        netcdf=results, var_name='ionisation_data_file')[0],
        encoding='utf-8').replace(' ', '')

    tot_imp_n = return_attribute(
        netcdf=results, var_name='total_impurity_density')[0]
    el_dens = return_attribute(
        netcdf=results, var_name='electron_density')[0]
    el_temp = return_attribute(
        netcdf=results, var_name='electron_temperature')[0]

    # Add Corona Result
    prb = str(return_attribute(
        netcdf=results, var_name='continuum_rad_data_file')[0],
        encoding='utf-8').replace(' ', '')

    plt = str(return_attribute(
        netcdf=results, var_name='tot_line_rad_data_file')[0],
        encoding='utf-8').replace(' ', '')

    data = Z((n_data, ny_tot))
    data[:, :nlin + 4] = imp_rad  # fills up until nlin + 3
    data[:, nlin + 4] = data[:, nlin + 3] - data[:, nlin]
    data[:, nlin + 5] = corona_radiation(
        rec_file=recomb, ion_file=ion, prb_file=prb, plt_file=plt,
        n_imp=tot_imp_n, density=el_dens, temperature=el_temp,
        debug=False)[0]

    if debug:
        print('data=', np.shape(data),
              '\ndata[:, 0:', nlin + 3, ']=', np.shape(imp_rad[:, :nlin + 3]),
              imp_rad[:, :nlin + 3],
              '\ndata[:, ', nlin + 4, ']=', np.shape(
                  data[:, nlin + 3] - data[:, nlin]),
              data[:, nlin + 3] - data[:, nlin],
              '\ndata[:, ', nlin + 5, ']=', np.shape(data[:, nlin + 5]),
              data[:, nlin + 5])

    return (data, grid_data, imp_rad, results, run_id)


def corona_radiation(
        rec_file='acd96_c.dat',
        ion_file='scd96_c.dat',
        prb_file='prb96_c.dat',
        plt_file='prb96_c.dat',
        n_imp=Z((1000, 200)),
        temperature=Z(1000),
        density=Z(1000),
        n_elements=1,
        debug=False):
    """ scaling and calculating the corona equilibrium radiation
    Keyword Arguments:
        rec_file {str} -- reombination rates file (default: {'acd96_c.dat'})
        ion_file {str} -- ionisation rates file (default: {'scd96_c.dat'})
        prb_file {str} -- bremsstrahlung rate file (default: {'prb96_c.dat'})
        plt_file {str} -- rates file (default: {'prb96_c.dat'})
        n_imp {[type]} -- number of impurities (default: {Z((1000, 200))})
        temperature {[type]} -- temperature vector (default: {Z(1000)})
        density {[type]} -- density vector (default: {Z(1000)})
        n_elements {int} -- number of elements (default: {1})
        debug {bool} -- debugging (default: {False})
    Returns:
        total {} -- total radiation calculations with all rates
        raten {} -- rates loaded and used
    """
    # create total radiation array
    n_te = len(temperature)
    total = Z((n_te))

    # create array for proton loss
    # add_hydrogen = 0
    # if keyword_set(add_h_continua) then begin
    #   p_loss = dblarr(nt)
    #   add_hydrogen = 1
    #   prb_h = add_h_continua
    # endif

    # logarithm of temperature and density
    temperature = np.log10(temperature)
    density = np.log10(density)

    # number of elements
    if not isinstance(rec_file, list):
        rec_file, ion_file, prb_file, plt_file = \
            [rec_file], [ion_file], [prb_file], [plt_file]
    n_el = np.shape(rec_file)[0]

    if debug:
        print('n_imp=', np.shape(n_imp), n_imp,
              '\nn_te=', n_te)

    # loop over different elements
    for j in range(n_el):
        # Corona-Distribution
        raten, n_meta = scdacd_corona(
            acd_file=rec_file[j], scd_file=ion_file[j],
            temp=temperature, dens=density, debug=False)

        if debug:
            print('raten=', np.shape(raten), raten,
                  '\nn_meta=', n_meta)

        # calculate proton loss
        # if add_hydrogen then begin
        #   charge = dindgen(n_meta+1)
        #   ff = rn#charge
        #   p_loss = p_loss + n_imp(*,i_el)*ff
        #   if (min(density-p_loss) lt 0.) then begin
        #     add_hydrogen = 0
        #     message,'Negative proton density! ' + $
        #             Calculation of H continuum stopped!',$
        #        /inform
        #   endif
        # endif

        # read line radiation data
        r_plt, t_plt, d_plt = read_atomdat(plt_file[j])[0:3]
        # add up line radiation
        for k in range(n_meta):
            a = bilin_interpol(
                matrix=r_plt[k, :, :], x_values=d_plt, y_values=t_plt,
                x_out=density, y_out=temperature)
            f = n_imp * raten[:, k] * 10 ** (a + density)
            total += f

            if debug:
                print('j / n_meta=', k, '/', n_meta,
                      '\na=', np.shape(a), a,
                      '\nf=', np.shape(f), f)

        if debug:
            print('plt tot=', np.shape(total), total)

        # read continuum radiation data
        r_prb, t_prb, d_prb = read_atomdat(prb_file[j])[0:3]
        # add up continuum radiation
        # for j=1,n_meta do begin
        for k in range(1, n_meta + 1):
            a = bilin_interpol(
                matrix=r_prb[k - 1, :, :], x_values=d_prb, y_values=t_prb,
                x_out=density, y_out=temperature)
            f = n_imp * raten[:, k] * 10 ** (a + density)
            total += f

            if debug:
                print('j / n_meta=', k, '/', n_meta,
                      '\na=', np.shape(a), a,
                      '\nf=', np.shape(f), f)

        if debug:
            print('prb tot=', np.shape(total), total)

    # add hydrogen continuum
    # if add_hydrogen then begin
    #   read_atomdat,prb_h,d,t,r,/verbose
    #   h_rad = (density-p_loss)*10^(bilin_interpol(r,d,t,dens,temp)+dens)
    #   tot = tot + h_rad
    # endif

    if debug:
        print('total=', np.shape(total), total,
              '\nraten=', np.shape(raten), raten)

    return (total, raten)


def scdacd_corona(
        acd_file='acd96_c.dat',
        scd_file='scd96_c.dat',
        qcd_file=None,
        temp=Z((1000)),
        dens=Z((1000)),
        META=False,
        debug=False):
    """ coronal equilibrium for ion stages
    Keyword Arguments:
        acd_file {str} -- rate file (default: {'acd96_c.dat'})
        scd_file {str} -- rate file (default: {'scd96_c.dat'})
        qcd_file {[type]} -- rate file (default: {None})
        temp {[type]} -- temperature vetcor (default: {Z((1000))})
        dens {[type]} -- density vector (default: {Z((1000))})
        META {bool} -- metastables? (default: {False})
        debug {bool} -- debugging (default: {False})
    Returns:
        fz {ndarray} -- coronal equilibrium data
        n_meta {int} -- number of metastables
    Notes:
        temp and dens are decadic logarithms of Te(eV), ne(cm-3)
    """
    if debug:
        print('dens=', dens, '\ntemp=', temp)

    n_te = temp.shape[0]

    # read ionisation data
    if not META:
        r_scd, t_scd, n_scd, n_meta = read_atomdat(file=scd_file)[0:4]
        n_trans = n_meta
        # charge = np.linspace(0, n_meta, n_meta + 1)
    elif META:
        # read_scd_r,scd_file,$
        #     d,t,r,n_trans,n_meta,start_ion,final_ion,charge,/verbose
        pass

    # declare arrays
    ion_rate = Z((n_te, n_trans))
    recomb_rate = Z((n_te, n_trans))

    if debug:
        print('n_scd=', n_scd, '\n', 't_scd=', t_scd)

    for j in range(n_trans):
        if not META:
            # s(*,j) = bilin_interpol(r(*,*,j),d,t,dens,temp)
            ion_rate[:, j] = bilin_interpol(
                matrix=r_scd[j, :, :], x_values=n_scd, y_values=t_scd,
                x_out=dens, y_out=temp)
            if debug:
                print('meta not set, r_scd[', j, ', :, :]=', r_scd[j, :, :],
                      '\nion_rate[:,', j, ']=', ion_rate[:, j])

        elif META:
            # s(*,j) = 10^(bilin_interpol(r(*,*,j),d,t,dens,temp))
            ion_rate[:, j] = 10 ** (bilin_interpol(
                matrix=r_scd[j, :, :], x_values=n_scd, y_values=t_scd,
                x_out=dens, y_out=temp))
            if debug:
                print('meta set, r_scd[', j, ', :, :]=', r_scd[j, :, :],
                      '\nion_rate[:,', j, ']=', ion_rate[:, j])

    # read recombination data
    if not META:
        r_acd, t_acd, n_acd = read_atomdat(file=acd_file)[0:3]
    elif META:
        # read_scd_r,acd_file,d,t,r,/verbose
        pass

    if debug:
        print('n_acd=', n_acd, '\n', 't_acd=', t_acd)

    for j in range(n_trans):
        if not META:
            #  al(*,j) = bilin_interpol(r(*,*,j),d,t,dens,temp)
            recomb_rate[:, j] = bilin_interpol(
                matrix=r_acd[j, :, :], x_values=n_acd, y_values=t_acd,
                x_out=dens, y_out=temp)
            if debug:
                print('meta not set, r_acd[', j, ', :, :]=', r_acd[j, :, :],
                      '\nrecomb_rate[:,', j, ']=', recomb_rate[:, j])

        elif META:
            #  al(*,j) = 10^(bilin_interpol(r(*,*,j),d,t,dens,temp)
            recomb_rate[:, j] = 10 ** (bilin_interpol(
                matrix=r_acd[j, :, :], x_values=n_acd, y_values=t_acd,
                x_out=dens, y_out=temp))
            if debug:
                print('meta set, r_acd[', j, ', :, :]=', r_acd[j, :, :],
                      '\nrecomb_rate[:,', j, ']=', recomb_rate[:, j])

    # read rates for transition between metastables of one ion
    if META:
        #  read_qcd_r,qcd_file,d,t,r,$
        #      n_trans,n_meta,start_meta,final_meta,/verbose
        #  qc = fltarr(n_temp,n_trans)
        #  for j=0,n_trans - 1 do
        #      qc(*,j) = 10^(bilin_interpol(r(*,*,j),d,t,dens,temp))
        pass

    # figure out korona ion distribution
    if not META:
        # fz = korona(s,al,n_meta)
        fz = corona(ior=ion_rate, recr=recomb_rate,
                    n_ion=n_meta, n_te=n_te, debug=False)[0]
        if debug:
            print('meta not set, fz=', np.shape(fz), fz)

    elif META:
        # fz = korona_r(
        #     s,al,qc,start_ion,final_ion,start_meta,final_meta,n_meta)
        pass

    return (fz, n_meta)


def corona(
        ior=Z((100, 100)),
        recr=Z((100, 100)),
        n_ion=10,
        n_te=1000,
        debug=False):
    """ calculates coronal eqilibrium from radiaton and rates given
    Keyword Arguments:
        ior {[type]} -- ionistaion rates (default: {Z((100, 100))})
        recr {[type]} -- recombination rates (default: {Z((100, 100))})
        n_ion {int} -- number of ions (default: {10})
        n_te {int} -- length of ne/te (default: {1000})
        debug {bool} -- debugging (default: {False})
    Returns:
        fz {ndarray} -- coronal equilibrium
        ff {ndarray} -- ion stage sum
    Notes:
        Figure out korona equilibrium. fz = nz/Sum(nz)
        -> ior:  log10 of ionisation rate(temperature,ionisation stage)
        -> recr: log10 of recombination rate(temperature,ionisation stage)
    """
    fz = Z((n_te, n_ion + 1))
    ff = Z((n_te))

    fz[0:n_te, 0] = 1.0

    if debug:
        print('fz=', np.shape(fz), fz)

    for i in range(1, n_ion + 1):
        fz[:, i] = fz[:, i - 1] * 10 ** (ior[:, i - 1] - recr[:, i - 1])

        if debug:
            print('ior[:,', i - 1, ']=', ior[:, i - 1],
                  '\nrecr[:,', i - 1, ']=', recr[:, i - 1],
                  '\nfz[:,', i - 1, ']', fz[:, i - 1],
                  '\nfz[:,', i, ']=', fz[:, i])

        f = fz[0:n_te, 0:i + 1]
        ff = np.sum(f, 1)
        if debug:
            print('f=', n_te, i + 1, np.shape(f),
                  '\nff=', np.shape(ff), ff)

        for j in range(i + 1):
            fz[:, j] = fz[:, j] / ff

    return (fz, ff)


def bilin_interpol(
        x_values=Z((1000)),
        y_values=Z((1000)),
        x_out=Z((500)),
        y_out=Z((500)),
        matrix=Z((1000, 1000)),
        method='bivariat_spline',
        MISSING=False,
        debug=False):
    """ bilinear interpolation routine
    Keyword Arguments:
        x_values {[type]} -- dimension (default: {Z((1000))})
        y_values {[type]} -- values (default: {Z((1000))})
        x_out {[type]} -- target dimension (default: {Z((500))})
        y_out {[type]} -- target values (default: {Z((500))})
        matrix {[type]} -- interpolation matrix (default: {Z((1000, 1000))})
        method {str} -- method to use (default: {'bivariat_spline'})
        MISSING {bool} -- what to do on missing dimension (default: {False})
        debug {bool} -- debugging (default: {False})
    Returns:
        out {ndarray} -- bilinear interpolation from old to new values/dim.
    """

    """
    function bilin_interpol, matrix         ,$ ; <--
                             x_values       ,$ ; <--
                             y_values       ,$ ; <--
                             xpout          ,$ ; <--
                             ypout          ,$ ; <--
                             grid=grid      ,$ ; <--
                             cubic = cubic  ,$ ; <--
                             missing=missing; <--
    """
    # Bilineare Interpolation einer Matrix
    if debug:
        print('y_out', np.shape(y_out), y_out)
        print('x_out', np.shape(x_out), x_out)
        print('y_values', np.shape(y_values), y_values)
        print('x_values', np.shape(x_values), x_values)

    X = interp1d(
        y=np.linspace(0, len(x_values) - 1, len(x_values)),
        x=x_values, fill_value='extrapolate')
    x_inter = X(x_out)
    Y = interp1d(
        y=np.linspace(0, len(y_values) - 1, len(y_values)),
        x=y_values, fill_value='extrapolate')
    y_inter = Y(y_out)

    if debug:
        print('y_inter', np.shape(y_inter), y_inter)
        print('x_inter', np.shape(x_inter), x_inter)
        print('matrix', np.shape(matrix), matrix)

    if not MISSING:
        if method == 'interpolate':
            f = interp2d(x=x_values, y=y_values, z=matrix, kind='cubic')
            M = f(x_inter, y_inter)
            out = M.diagonal()[::-1]

        elif method == 'grid':
            xx, yy = np.meshgrid(x_values, y_values)
            xnew, ynew, z = xx.ravel(), yy.ravel(), matrix.ravel()
            M = griddata((xnew, ynew), z,
                         (x_inter[None, :], y_inter[:, None]),
                         method='cubic')
            out = M.diagonal()[::-1]

        elif method == 'bivariat_spline':
            f = RBS(x=x_values, y=y_values, z=matrix.transpose())
            M = out = f.ev(xi=x_out, yi=y_out)

    elif MISSING:
        f = interp2d(x=x_inter, y=y_inter, z=matrix,
                     kind='cubic', fill_value=0.0)
        M = f(x_inter, y_inter)
        out = M.diagonal()[::-1]

    if debug:
        print('M=', np.shape(M), M,
              '\nout=', np.shape(out), out)

    return (out)


def period_system(
        charge=1,
        symbol=None,
        debug=False):
    """ returns period table info on elements
    Keyword Arguments:
        charge {int} -- elemental charge/electrom count (default: {1})
        symbol {[type]} -- element symbol (PSE) (default: {None})
        debug {bool} -- debugging (default: {False})
    Returns:
        symbol {[str]} -- element symbols (PSE)
        charge {[int]} -- element charges (electorn count)
        mass {[float]} -- element masses
        density {[float]} -- volumen densities
        number_density {[float]} -- number densities
    """
    if (charge is not None) & (symbol is None):
        if not isinstance(charge, list) and \
                not isinstance(charge, np.ndarray):
            charge = [charge]
        L = len(charge)
        name, mass, symbol, density, number_density, elements = \
            [None] * L, [None] * L, [None] * L, \
            [None] * L, [None] * L, [None] * L

        for i, c in enumerate(charge):
            try:
                elements[i] = period.elements[int(c)]

            except Exception:
                if debug:
                    print('\\\ couldnt find element with charge ', c, int(c))

    elif (charge is None) & (symbol is not None):
        if not isinstance(symbol, list) and \
                not isinstance(charge, np.ndarray):
            symbol = [symbol]
        name, mass, charge, density, number_density, elements = \
            [None] * L, [None] * L, [None] * L, \
            [None] * L, [None] * L, [None] * L

        for i, s in enumerate(symbol):
            try:
                elements[i] = period.elements.__dict__[s]

            except Exception:
                if debug:
                    print('\\\ couldnt find element with symbol ', s)

    for i, element in enumerate(elements):
        try:
            name[i] = element.name
            mass[i] = element.mass
            symbol[i] = element.symbol

            if element.density_units == 'g/cm^3':
                density[i] = element.density * 1e3
            else:
                if debug:
                    print('\\\ ', element, 'has density',
                          element.density_units)
                density[i] = element.density

            if element.number_density_units == '1/cm^3':
                number_density[i] = element.number_density * 1e6
            else:
                if debug:
                    print('\\\ ', element, 'has number density',
                          element.number_density_units)
                number_density[i] = element.number_density

        except Exception:
            if debug:
                print('\\\ couldnt find property of element', element)

    if debug:
        print('symbol=', symbol.__class__, symbol,
              '\ncharge=', charge.__class__, charge,
              '\nmass=', mass.__class__, mass,
              '\ndensity=', density.__class__, density,
              '\nnumber_density=', number_density.__class__, number_density)

    return (symbol, charge, mass, density, number_density)


def strahl_integral(
        y=Z((10, 1000)),
        netcdf=None,
        lin=False,
        rad=False,
        vol=False,
        area=False,
        weight=None,
        wdata_weight=None,
        rho_range=None,
        average=False):
    """ line and volume integral of STRAHL data
    Keyword Arguments:
        y {ndarray} -- data array (default: {Z((10, 1000))})
        netcdf {netcdf object} -- results loaded from file (default: {None})
        lin {bool} -- do line integral? (default: {False})
        rad {bool} -- radial integration? (default: {False})
        vol {bool} -- do volume integral? (default: {False})
        area {bool} -- do area integral? (default: {False})
        weight {[type]} -- weights to use (default: {None})
        wdata_weight {[type]} -- other weights possible (default: {None})
        rho_range {[type]} -- poloidal rho range (default: {None})
        average {bool} -- averaging? (default: {False})
    Returns:
        integral {ndarray} -- volume/line integrated data returned
    """
    s = [len(np.shape(y))]
    for dim in np.shape(y):
        s.append(dim)

    if rad:  # keyword_set(rad) then begin
        p = return_attribute(netcdf=netcdf, var_name='pro')[0]
        rr = return_attribute(netcdf=netcdf, var_name='radius_grid')[0]

        ff = .5 / p
        ff[0] = rr[1] / 2.
        ff[1] = .5 * ff[1] + ff[0]

    if lin:
        ff = return_attribute(netcdf=netcdf, var_name='drm')[0]

    if vol:
        p = return_attribute(netcdf=netcdf, var_name='pro')[0]
        rr = return_attribute(netcdf=netcdf, var_name='radius_grid')[0]
        r_maj = return_attribute(netcdf=netcdf, var_name='large_radius')[0]
        circum = 2 * np.pi * r_maj
        ff = circum * np.pi * rr / p

    if area:
        p = return_attribute(netcdf=netcdf, var_name='pro')[0]
        rr = return_attribute(netcdf=netcdf, var_name='radius_grid')[0]
        ff = np.pi * rr / p

    if weight is not None:
        if wdata_weight is None:
            ff_w = return_attribute(netcdf=netcdf, var_name=weight)[0]
        elif wdata_weight is not None:
            ff_w = wdata_weight

        sw = np.shape(ff_w)
        if sw[0] == 1:
            ff = ff * ff_w
        if sw[0] == 2 and s[0] == 2:
            y = y * ff_w
        if sw[0] == 2 and s[0] == 3:
            if sw[1] == s[1] and sw[2] == s[2]:
                for i in range(0, s[3]):
                    y[:, :, i] = y[:, :, i] * ff_w
            if sw[1] == s[1] and sw[2] == s[3]:
                for i in range(0, s[2]):
                    y[:, i, :] = y[:, i, :] * ff_w

    rho = return_attribute(netcdf=netcdf, var_name='rho_poloidal_grid')[0]
    if rho_range is not None:
        ff = ff[(rho >= rho_range[0]) & (rho <= rho_range[-1])]
        y = y[(rho >= rho_range[0]) & (rho <= rho_range[-1])]

    if vol:
        nf = np.shape(ff)[0]
        ff[nf - 1] = 0.125 * ff[nf - 1]
        ff[nf - 2] = 0.875 * ff[nf - 2]

    if rad:
        nf = np.shape(ff)[0]
        ff[nf - 1] = 0.125 * ff[nf - 1]
        ff[nf - 2] = 0.875 * ff[nf - 2]

    if s[0] == 1 or s[0] == 2:
        integral = np.dot(np.transpose(ff), y)
    elif s[0] == 3:
        integral = Z((s[2], s[3]))
        for i in range(0, s[3]):
            integral[:, i] = np.dot(np.transpose(ff), y[:, :, i])

    if average:
        y = y - y + 1.
        s = np.shape(y)

        if s[0] <= 2:
            norm = np.dot(np.transpose(ff), y)
        elif s[0] == 3:
            norm = Z((s[2], s[3]))
            for i in range(0, s[3]):
                norm[:, i] = np.dot(np.transpose(ff), y[:, :, i])

        integral *= 1 / norm

    s = np.shape(integral)

    return (integral)


def read_atomdat(
        file='prb96_c.dat',
        debug=False):
    """
    Keyword Arguments:
        file {str} -- (default: {'prb96_c.dat'})
        debug {bool} -- debugging (default: {False})
    Returns:
        density {ndarray} -- density grid
        temperature {ndarray} -- temperature grid
        rates {ndarray} -- rate coefficients (density,temperature,ion)
        n_ions {int} -- number of ions
        n_ne {int} -- number of density points
        n_te {int} -- number of temperature points
    """

    loc = '../../strahl/atomdat/newdat/' + file

    read_in = pd.read_csv(loc).values
    header = pd.read_csv(loc, nrows=1).keys()[0]
    n_ions, n_ne, n_te = [int(i) for i in header.split() if i.isdigit()][0:3]

    #  readf,u,format='(8(f10.5))',density, in cm^-3, log10
    density, M = Z((n_ne)), len(read_in[1][0].split())
    L = int(np.ceil(n_ne / 8))
    for i, row in enumerate(read_in[1:L + 1]):
        for j, obj in enumerate([float(i) for i in row[0].split()]):
            density[(i * M) + j] = obj
            if debug:
                print(i, j, (i * M) + j, '\t', obj)
    if debug:
        print('\n')

    # readf,u,format='(8(f10.5))',temperature
    temperature, N = Z((n_te)), len(read_in[2][0].split())
    K = int(np.ceil(n_te / 8))
    for i, row in enumerate(read_in[L + 1:K + L + 1]):
        for j, obj in enumerate([float(i) for i in row[0].split()]):
            temperature[(i * N) + j] = obj
            if debug:
                print(i, j, (i * N) + j, '\t', obj)
    if debug:
        print('\n')

    rates = Z((n_ions, n_te, n_ne))
    for i in range(n_ions):
        for j in range(n_te):
            G = (K + L + 1) + (L * i * n_te) + L * j + 1 + i

            for k, obj in enumerate(
                    [float(i) for i in read_in[G][0].split()]):
                if debug:
                    print(i, j, G, '\t', obj)
                rates[i, j, k] = obj

    return (rates, temperature, density, n_ions, n_te, n_ne)
