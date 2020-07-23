import pytest
import os
import sys

""" Syspath needs to include parent directory "pollen classification" and "Code" to find sibling 
modules and database."""
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
sys.path.append(file_path)
from config import util_ops


@pytest.mark.parametrize('iterable, result',
                         [
                             ([1,2,3,4], 24),
                             ((-1,-2,3,4), 24)
                         ])
def test_multiply(iterable, result):
    assert util_ops.multiply(iterable) == result


