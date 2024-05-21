""" **************************************************************************
    so header """

import warnings
import numpy as np
import json
import versioning as vers
import urllib

import mClass
from mClass import query_yes_no

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

""" eo header
************************************************************************** """


def upload_archive(
        links=['_DATASTREAM', '_PARLOG'],
        data={'values': None, 'dimensions': None},
        parms={'Unit': None, 'dimensionSize': None},
        base_URI='http://archive-webapi.ipp-hgw.mpg.de/Sandbox/' +
                 'raw/W7XAnalysis/',
        indent_level='\t',
        header={"Content-Type": "application/json"},
        debug=True):
    """Uploads the to the archive
    Args:
        links (1, list): DATASTREAM and PARLOG links in that order
        data (2, dict): Dictionary of data to post
        parms (3, dict): Parameter block to upload
        base_URI (0, str): Where to write in the archive
        indent_level (5, str): Indentation level
        header (4, dict): Request archive header to get data
        debug (bool, optional): Description
    Returns:
        dat_stat (0, str): Saying whether successful or not
        par_stat (1, str): -- " --
    Notes:
        None.
    """
    [url_data, url_param] = \
        [base_URI + x for x in [links[0], links[1]]]
    """ check the version on the links to write """
    [max_version, max_code_release, last_analysis_env] = \
        vers.get_versions_list(
            location=url_data + '_versions.json',
            base_URI=base_URI, indent_level=indent_level)
    loc = url_data.replace(base_URI, '')

    if (max_version == 0):
        query = '>> No vers found at ' + loc
        foo = query_yes_no(
            indent_level + query + ', continue?', None)
        if not foo:
            return ['FAIL', 'FAIL']

    elif (max_version >= 1):
        query = '>> Vers V' + str(max_version) + ' at ' + loc
        [url_data, url_param] = \
            [x + 'V' + str(max_version)  # + '/'
             for x in [url_data, url_param]]
    if debug:
        print(indent_level + query)

    """ set up dictionaries for json'ing """
    parms_js = json.dumps(mClass.dict_transf(
        parms, to_list=True))

    data_js = json.dumps(mClass.dict_transf(
        data, to_list=True))

    data_req = urllib.request.Request(
        url_data, data=data_js.encode("utf-8"), headers=header)
    param_req = urllib.request.Request(
        url_param, data=parms_js.encode("utf-8"), headers=header)

    """ finally make POST requests and send stuff to archive """
    indent_level += '\t'
    try:
        data_resp = urllib.request.urlopen(data_req)
        if debug:
            print(indent_level + "Multichan data resp:\t", data_resp)
        dat_stat = 'GREEN'
    except urllib.error.HTTPError:
        if debug:
            print(indent_level +
                  "Multichan data HTTP: Bad request 400...")
            print(indent_level + "Multichan data req:\t", data_req)
        dat_stat = 'FAIL'
    try:
        param_resp = urllib.request.urlopen(param_req)
        if debug:
            print(indent_level + "Multichan PARAM resp:\t", param_resp)
        par_stat = 'GREEN'
    except urllib.error.HTTPError:
        if debug:
            print(indent_level +
                  "Multichan PARAM URLLIB HTTP: Bad request 400...")
            print(indent_level + "Multichan PARAM req:\t", param_req)
        par_stat = 'FAIL'

    return [dat_stat, par_stat]
