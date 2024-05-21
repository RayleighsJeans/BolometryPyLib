import argparse
from newton import converter
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='newton',
        description='mass to weight')

    # float of mass
    parser.add_argument(
        '--mass',
        '-m',
        type=float,
        help='mass in kg',
        required=False)

    # weight
    parser.add_argument(
        '--weight',
        '-w',
        type=float,
        help='weigt in N',
        required=False)

    # name
    parser.add_argument(
        '--name',
        '-n',
        type=str,
        required=False)

    # g
    parser.add_argument(
        '--gravity',
        '-g',
        type=float,
        help='gravitational class',
        required=False,
        default=9.81)

    args = parser.parse_args()
    if args.name != 'None':
        print(args.name)

    if args.mass != None:
        print('calculate weight from mass: ', end='')
        weight = converter.mass_to_weight(args.mass, args.gravity)
        print('%s N' % weight)

    if args.weight != None:
        print('calculate mass from weight: ', end='')
        mass = converter.weight_to_mass(args.weight, args.gravity)
        print('%s kg' % mass)

    return
