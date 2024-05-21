""" **************************************************************************
    start of file """

import heapq
import numpy as np

""" eo ******************************************************************* """


class mClass(object):
    def __init__(self):
        self.default = None


def Roman_to_int(
        roman,
        debug=False):

    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1]

    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"]

    integer = 0
    for i, symbol in enumerate(roman):
        if debug:
            print(len(roman), roman, i, ': ',
                  symbol, val[syb.index(symbol)])
        if i < len(roman) - 1:
            if val[syb.index(symbol)] < val[syb.index(roman[i + 1])]:
                integer -= val[syb.index(symbol)]
            else:
                integer += val[syb.index(symbol)]
        else:
            integer += val[syb.index(symbol)]
    return integer


def int_to_Roman(
        num,
        debug=False):

    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1]

    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"]

    roman_num = ''
    i = 0

    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num


def second_smallest(numbers):
    return heapq.nsmallest(2, numbers)[-1]


def second_largest(numbers):
    return heapq.nlargest(2, numbers)[-1]


def debug(parser, parsername):

    try:
        size = np.shape(parser)
    except TypeError:
        # print('Error:', repr(e))
        try:
            size = len(parser)
        except TypeError as e:
            print('Error:', repr(e))
            return

    max_length = max(size)
    if (isinstance(parser, dict)):
        for key, value in parser.items():

            if (isinstance(key, dict)):
                for key2, value2 in key.items():
                    if (isinstance(key2, dict)):
                        for key3, value3 in key2.items():
                            if max_length < size(value3):
                                max_length = size(value3)
                    else:
                        if max_length < size(value2):
                            max_length = size(value2)
            else:
                if max_length < size(value):
                    max_length = size(value)

    if (max_length <= 100):
        print('name:', parsername, 'type: ', type(parser),
              'size: ', size, 'content:', parser)
    else:
        if (isinstance(parser, dict)):
            keylist = []
            for key, value in parser.items():
                keylist.append(key)
            print('name:', parsername, 'type: ', type(parser),
                  'size: ', size, 'content:', parser,
                  'keylist: ', keylist)
        elif (isinstance(parser, list)):
            print('name:', parsername, 'type: ', type(parser),
                  'size: ', size, 'content:', parser[0:100], ' ...')
        elif (isinstance(parser, np.ndarray)):
            print('name:', parsername, 'type: ', type(parser),
                  'size: ', size, 'content:', parser[0:10][0:10], ' ...')

    return


def void_f():
    while True:
        pass

    return


def find_nearest(array, value):
    if isinstance(array, np.ndarray):
        idx = (np.abs(array - value)).argmin()

    elif isinstance(array, list):
        idx = (np.abs([x - value for x in array])).argmin()

    return idx, array[idx]


def dict_transf(
        dictionary={'none': {'none': []}},
        to_list=False,  # to array
        verbose=False):  # silent

    def print_type(key='none', value=None):
        print(key, 'type:', value.__class__)
        return

    def list_array_mod(value):

        if isinstance(value, dict):
            for key, V in value.items():

                if isinstance(V, np.ndarray):
                    if verbose:
                        print_type(key, V)

                    if V.size > 0:
                        if isinstance(V[0], dict):
                            for i, v in enumerate(V):
                                V[i] = list_array_mod(v)

                    if to_list:
                        value[key] = V.tolist()

                if isinstance(V, list):
                    if verbose:
                        print_type(key, V)

                    if V != []:
                        if isinstance(V[0], dict):
                            for i, v in enumerate(V):
                                V[i] = list_array_mod(v)

                    if not to_list:
                        if V != []:
                            dtype = 'float' if len(
                                np.shape(V)) > 1 else type(V[0])
                        else:
                            dtype = 'float'

                        value[key] = np.array(
                            V, dtype=dtype)

                if isinstance(V, dict):
                    value[key] = list_array_mod(V)

        else:

            if isinstance(value, np.ndarray):

                if value.size > 0:
                    if isinstance(value[0], dict):
                        for i, V in enumerate(value):
                            value[i] = list_array_mod(V)

                if verbose:
                    print_type(value)
                if list:
                    value = value.tolist()

            if isinstance(value, list):
                if value != []:
                    if isinstance(value[0], dict):
                        for i, V in enumerate(value):
                            value[i] = list_array_mod(V)

                if verbose:
                    print_type()(value)
                if not to_list:
                    if V != []:
                        dtype = 'float' if len(
                            np.shape(V)) > 1 else type(V[0])
                    else:
                        dtype = 'float'

                    value = np.array(
                        value, dtype=dtype)

        return (value)

    if str(type(dictionary)) == '<class \'dict\'>':
        pass
    else:
        print('wrong type:', type(dictionary))
        return

    try:
        list_array_mod(dictionary)
    except AttributeError as lvl:
        if prints:
            print('error:', repr(lvl))

    return dictionary


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        choice = input(question + prompt + ' ')
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            choice = input("Please respond with 'yes' or 'no' ('y'/'n'). ")
