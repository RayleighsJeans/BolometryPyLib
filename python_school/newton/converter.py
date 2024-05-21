import sys


def mass_to_weight(mass, g):
    try:
        assert mass >= 0
        weight = round(mass * g, 2)
    except Exception as e:
        return
        print('no negative mass', repr(e))
        sys.exit(1)
    else:
        return round(weight, 2)


def weight_to_mass(weight, g):
    try:
        assert weight >= 0
        mass = round(weight / g, 2)
    except Exception as e:
        return
        print('no negative weight', repr(e))
        sys.exit(1)
    else:
        return round(mass, 2)
