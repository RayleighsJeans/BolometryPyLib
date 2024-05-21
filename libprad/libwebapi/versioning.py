""" **************************************************************************
    so header """

import warnings
import requests
import numpy as np
import json
import urllib
from mClass import query_yes_no

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

""" eo header
************************************************************************** """


def create_version_link(
        user='pih',
        reason='Test',
        code_release='v0.1',
        environment='PC-CLD-WS-535',
        URL='none',
        indent_level='\t'):
    """ Creates a new version/a version at all for a given link in the
        W7X archive.
    Args:
        user (0, str): IPP net user name
        reason (1, str): Change in code or eval reason.
        code_release (2, str): vX.Y odering scheme.
        environment (3, str): Development environment.
        URL (4, str): Link to place where to create given record.
        indent_level (5, str): Printing indentation level.
    Returns:
        None.
    Notes:
        None.
    """

    header = {"Content-Type": "application/json"}
    version_info = \
        {'versionInfo': [
         {"reason": reason,
          "producer": user,
          "code_release": code_release,
          "analysis_environment": environment
          }]}

    status = True
    """ create needed json version object with header """
    vers_js = json.dumps(version_info)
    """ urllib requests at given url (POST) """
    version_req = \
        urllib.request.Request(URL, data=vers_js.encode("utf-8"),
                               headers=header)

    """ reduce printing """
    URL = URL.replace('http://archive-webapi.ipp-hgw.mpg.de/' +
                      'Test/raw/W7XAnalysis/', '')  # clear link
    URL = URL.replace('_DATASTREAM', '')  # clear suffix
    URL = URL.replace('_PARLOG', '')  # or this
    failed = URL.replace('/_versions.json', '')

    try:  # upload and if error, print debug and results
        version_resp = urllib.request.urlopen(version_req)
    except Exception:
        status = False
    if not status:  # failed
        print(indent_level + '>> failed to versionize at' + failed)
    elif status:  # all cool
        print(indent_level + '>> successfully versionized at ' + failed +
              ' with: ' + code_release)

    if False:
        print(indent_level + '>> failed with ', version_resp)
    return


def get_versions_list(
        location='none',
        base_URI='http://archive-webapi.ipp-hgw.mpg.de/Sandbox/' +
                 'raw/W7XAnalysis/',
        indent_level='\t'):
    """ Grab version list for given location
    Args:
        location (0, str): link to look at
        indent_level (1, str): printing indentation
    Returns:
        number (0, int): number of V in link
        code_release (1, str): vX.Y of the link in code
    Notes:
        None.
    """
    [start, stop] = [0, -1]
    filter_query = {'filterstart': start, 'filterstop': stop}
    headers = {'Accept': 'application/json'}
    versions_req = requests.get(location, headers=headers,
                                params=filter_query)
    versions = versions_req.json()
    location = location.replace(base_URI, '')
    try:
        L = len(versions['versionInfo']) - 1
        return [versions['versionInfo'][L]['number'],
                versions['versionInfo'][L]['code_release'],
                versions['versionInfo'][L]['analysis_environment']]
    except Exception:
        try:
            if (versions['status'] == 400):
                message = versions['message']
                print(indent_level + '>> No versioning existing for ' +
                      location)
            else:
                print(indent_level +
                      '>> different status returned ', versions['status'],
                      '\n' + indent_level + '>> message: ' + message)
        except Exception:
            pass
            print(indent_level +
                  '>> No version returned at ' + location)
        return [0, 'v0.0', 'None']


def synchronize(
        URL='none',
        vers=1,
        URL_par='none',
        vers_par=1,
        indent_level='\t'):
    """ Synchronizing the links found of parlog and datastream
    Args:
        URL (0, str): link
        vers (1, int): version number
        URL_par (2, str): parlog link
        vers_par (3, int): parlog version number
        indent_level (4, str): indentation printing
    Returns:
        None.
    """
    if (vers > vers_par):
        [L_URI, S_URI, L_vers] = [URL, URL_par, vers_par]
    elif (vers_par >= vers):
        [L_URI, S_URI, L_vers] = [URL_par, URL, vers]
    [start, stop] = [0, -1]
    filter_query = {'filterstart': start, 'filterstop': stop}
    headers = {'Accept': 'application/json'}
    versions_req = requests.get(L_URI, headers=headers,
                                params=filter_query)
    versions = versions_req.json()
    final = versions['versionInfo'][-1]
    target = final['number']
    user = final['producer']
    code_release = final['code_release']
    environment = final['analysis_environment']
    reason = \
        final['reason'] + '; found out of synch log entries, versioning up'

    while L_vers < target:
        create_version_link(user, reason, code_release,
                            environment, S_URI, indent_level)
        L_vers += 1
    return


def increase_versioning(
        indent_level='\t',
        user='pih',
        reason='none',
        environment='PC-CLD-WS-535, Win7 Py3.6.X',
        links=['BoloAdjusted', 'PradChannels', 'PradHBCm', 'PradVBC',
               'MKappa', 'RKappa', 'MRes', 'RRes', 'MTau', 'RTau',
               'ChordalProfile_HBCm', 'ChordalProfile_VBC', 'LInHBCm',
               'LInVBCr', 'LInVBCl', 'LInSXRv', 'LInSXRh', 'LInAlch',
               'BoloSignal'],
        base='Test/raw/W7XAnalysis/',
        specifier='BoloTest5/'):
    """ writing to the archive and increasing the versioned streams
    Args:
        indent_level (str, optional): printing indentation
        user (str, optional): user
        reason (str, optional): why made update
        environment (str, optional): development envo
        links (list, optional): where to look, given the base
        specifier (str, optional): base in raw archive
    Returns:
        None
    """
    archives = ['_DATASTREAM/', '_PARLOG/']
    # where to look/do
    base0 = 'http://archive-webapi.ipp-hgw.mpg.de/'

    foo = False
    # going through all th links at the given parent loc
    for link in links:
        loc = base + specifier + link

        # check the version number and select hence worth
        URL = base0 + base + specifier + link + \
            "_DATASTREAM/" + '_versions.json'
        [max_version, max_code_release, last_analysis_env] = \
            get_versions_list(URL, indent_level)

        # found that possibly PARLOG and DATASTREAM might be out of sync
        URL_par = base0 + base + specifier + link + \
            "_PARLOG/" + '_versions.json'
        [max_version_par, max_code_release_par, last_analysis_env_par] = \
            get_versions_list(URL_par, indent_level)

        if (max_version != max_version_par):
            print(indent_level + '>> versions of PARLOG and DATASTREAM' +
                  ' at ' + loc + ' are out of sync')
            foo = query_yes_no(indent_level +
                               '>> synchronize Vers on ' + loc, None)
            if foo:
                synchronize(URL, max_version, URL_par,
                            max_version_par, indent_level)
                continue
            else:
                print(indent_level +
                      '>> Unsynchronized versions in log, leaving')
                return

        if (max_version == 0):
            query = '>> no vers at ' + loc + ', create vers?'
        elif (max_version == 1):
            query = '>> V1 found at ' + loc + ', nxt vers?'
        elif (max_version >= 1):
            query = '>> Ver V' + str(max_version) + ' at ' + loc + ', nxt?'
        else:
            print(indent_level +
                  '>> versioning exception not predicted ...')
            return
        foo = query_yes_no(indent_level + query, None)  # ask for action

        # for datastream and parlog
        if foo:  # should create new link
            code_release = 'v' + str(int(max_version + 1)) + '.0'
            for ark in archives:
                create_version_link(user=user, reason=reason,
                                    code_release=code_release,
                                    environment=environment,
                                    URL=base0 + base + specifier +
                                    link + ark +
                                    '_versions.json',
                                    indent_level=indent_level)
    return
