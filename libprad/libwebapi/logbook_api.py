""" **************************************************************************
    so header """

import warnings
import requests
import sys
import numpy as np
from pprint import pprint as pprint

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

# names in json format of logbook entrance to look for
infos = ['duration',
         'ECRH',
         'ECRH energy',
         'ECRH duration',
         'Main field',
         'Gas1',
         'Gas2',
         'Gas3',
         'Pellet']

""" eo header
************************************************************************** """


def db_request_lobgook(
        date='20181010',
        shot=32,
        debug=False):
    """ download entire or partial logbook api entry for date (+shot)
    Args:
        date (0, str, optional): XP date
        shot (1, int, optional): XP id
        debug (2, bool, optional): Debugging bool
    Returns:
        success (0, bool): got something?
        request (1, list): results found
    Notes:
        None.
    """
    try:
        if shot is not None:
            req = requests.get(
                url='https://w7x-logbook.ipp-hgw.mpg.de/api/' +
                'search.html?&q=id:XP_' + date + '.' + str(shot)).json()
            if debug:
                pprint(req['hits']['hits'][0], depth=2)
            return (True, req['hits']['hits'][0])

        elif shot is None:
            req = requests.get(
                url='https://w7x-logbook.ipp-hgw.mpg.de/api/' +
                'search.html?size=100&q=id:XP_' + date + '.*').json()
            if debug:
                pprint(req['hits'], depth=1)
            return (True, req['hits']['hits'])

    except Exception:
        return (False, None)


def db_request_component(
        date='20181010',
        id=32,
        component='QSQ',
        debug=False):
    """ download and return component log for experiment
    Args:
        date (0, str, required): XP date
        shot (1, int, required): XP id
        debug (2, bool, required): Debugging bool
    Returns:
        status (0, bool): status of return
        result (1, dict): component logs
    """
    if True:  # try:
        if id is not None:
            req = requests.get(
                url='https://w7x-logbook.ipp-hgw.mpg.de/api/log/' +
                component + '/XP_' + date + '.' + str(id)).json()
            if ('status_code' in req.keys()):  # failed
                return (False, None)

            val = [{'name': entry['name'],
                    'value': entry['value'],
                    'unit': entry['unit']}
                   for entry in req['_source']['tags']]

            res = {}
            for i, entry in enumerate(val):
                res[entry['name']] = {'value': entry['value'],
                                      'unit': entry['unit']}

            if debug:
                pprint(res)
            return (True, res)

        else:
            return (False, None)

    # except Exception:
    #     return (False, None)


def load_comments(
        date='20180920',
        shot=None,
        debug=False,
        indent_level='\t'):

    state, req = db_request_lobgook(
        date=date, shot=shot)
    if state:
        return (req['_source']['comments'])

    return ([None])


def filter_comments(
        date='20180920',
        shot=None,
        users=['viwi'],
        debug=False):

    comments = load_comments(
        date=date, shot=shot, debug=debug)
    if comments is not None:
        res = []
        for j, comment in enumerate(comments):
            if comment['user'] in users:
                res.append(comment['content'])
        return (res)
    return (None)


def db_greb_hit(
        req={'_id': 'XP_20181010.32',
             '_source': {
                 'comments': [],
                 'component_status': [],
                 'description': 'none',
                 'from': 1539179650333392501,
                 'id': 'XP_20181010.32',
                 'scenarios': [],
                 'tags': [],
                 'upto': 1539179729933392500},
             '_type': 'XP_logs'},
        tag='ECRH',
        spec='name',
        debug=False):
    """ get the entry you are looking for in the results
    Args:
        req (0, dict, optional): XP ID logbook entry
        tag (1, str, optional): Tag or partial name to grab
        debug (2, bool, optional): Debugging bool
    Returns:
        res (0, list): either literal ort list of partial comprehensions
    Notes:
        None.
    """
    taglist = [x[spec] for x in req['_source']['tags']]
    if debug:
        print('\ttaglist:', taglist)
    res = []
    if tag not in taglist:
        if debug:
            print('\ttag:', tag, 'not in literal comparison')
        for i, name in enumerate(taglist):
            if tag in name:
                if debug:
                    print('\ttag:', tag, 'partially in', name,
                          '\n\t\\\ ', req['_source']['tags'][i],
                          '\n', res)
                res.append(req['_source']['tags'][i])
    elif tag in taglist:
        if debug:
            print('\ttag:', tag, 'literal occurance',
                  '\n\t\\\ found:', np.where(np.array(taglist) == tag)[0])
        for i in np.where(np.array(taglist) == tag)[0]:  # multi hits
            res.append(req['_source']['tags'][i])

    return (res)


def logbook_json_load(
        lu_list=None,
        date='20181010',
        group='name',
        shot=None,
        debug=False,
        printing=False,
        indent_level='\t'):
    """ get the ID entry or day and put that in a list of {info}-values
    Args:
        date (0, str, optional): XP day
        shot (1, None, optional): XP id
        debug (2, bool, optional): Debugging bool
        indent_level (3, str, optional): Indentation printing level
    Returns:
        logbook_data (0, list): either full day or shot info, based off of
            {info}-list up top
    Notes:
        None.
    """
    if printing:
        print(indent_level + '>> Get info from logbook ...')

    if lu_list is None:
        lu_list = infos

    logbook_data = {}
    if shot is not None:
        for tag in lu_list:
            logbook_data[tag] = []
            hits = db_greb_hit(
                req=db_request_lobgook(
                    date=date, shot=shot, debug=debug)[1],
                tag=tag, debug=debug, spec=group)
            for hit in hits:
                logbook_data[tag].append(hit)

    elif shot is None:
        req = db_request_lobgook(date=date, shot=None)[1]

        xpids = [x for i, x in enumerate(req) if '2018' in x['_id']]
        lists = np.zeros((2, len(xpids)))

        for i, x in enumerate(xpids):
            lists[:, i] = i, int(x['_id'].replace('XP_', '').replace('.', ''))

        sort = lists[:, lists[1, :].argsort()]
        for j, i in enumerate(sort[0]):
            for tag in lu_list:
                logbook_data[tag] = []
                hits = db_greb_hit(req=req[int(i)], tag=tag, spec=group)
                for hit in hits:
                    logbook_data[tag].append(hit)

    return (logbook_data)  # hit_list


def get_info_of_shot(
        date='20181010',
        shot=32,
        debug=False,
        indent_level='\t',
        logbook_info=[]):
    """ loading from each tag hit the data and putting/appending it to info
    Args:
        date (0, str, optional): XP day
        shot (1, int, optional): XP id
        debug (2, bool, optional): Debugging bool
        indent_level (3, str, optional): Printing indentation level
        logbook_info (4, list, optional): List of tag/label hits, dicts
    Returns:
        info_of_shot/_day (0, list): List of detailed infos (numbs) for day
            or individual XP ids
    Notes:
        None.
    """
    info_of_shot = []
    print(indent_level + '>> get info of shot ' + str(shot))

    if shot is None:
        logbook_info = logbook_json_load(
            date=date, shot=shot, indent_level=indent_level)

        info_of_day = []
        for i, x in enumerate(logbook_info):
            info_of_day.append(shot_info(
                logbook_info=logbook_info,
                shot=i, indent_level=indent_level))

        return info_of_day

    elif shot is not None:
        info_of_shot = shot_info(
            logbook_info=logbook_info, shot=shot, indent_level=indent_level)
        return info_of_shot


def shot_info(
        logbook_info=[],
        shot=32,
        debug=False,
        indent_level='\t'):
    """ looky looky for iffy iffy
    Args:
        logbook_info (0, list, optional): List of dicts with tag hits
        shot (int, optional): Spot in list
        debug (bool, optional): Debugging bool
        indent_level (str, optional): Printing indentation level
    Returns:
        shot_info (0, list): Info list with values (numbs)
    """
    shot_info = []
    dat = logbook_info[shot]
    if dat != []:
        for i, x in enumerate(infos):
            try:
                shot_info.append(dat[i]['valueNumeric'])
            except Exception:
                try:
                    shot_info.append(dat[i]['value'])
                except Exception:
                    shot_info.append(None)

    return shot_info
