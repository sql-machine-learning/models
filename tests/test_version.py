import sqlflow_models
import unittest


def test_answer():
    assert sqlflow_models.__version__ == sqlflow_models._version.__version__