from newton.converter import mass_to_weight
from newton.converter import weight_to_mass
from nose.tools import raises

g = 9.81

def test_mass_to_weight():
    weight = mass_to_weight(10., g)
    assert weight == 98.1
    return

def test_weight_to_mass():
    mass = weight_to_mass(98.1, g)
    assert mass == 10.
    return

@raises(SystemExit)
def test_negative_mass():
    weight = mass_to_weight(-10., g)

@raises(SystemExit)
def test_negative_weight():
    mass = weight_to_mass(-98.1, g)
