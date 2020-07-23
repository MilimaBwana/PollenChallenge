import sys
import os
import pytest
import numpy as np
import sklearn.metrics

""" Syspath needs to include parent directory "pollen classification" and "Code" to find sibling 
modules. """
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
sys.path.append(file_path)

from nn_util import metrics


def test_precision_macro():
    y_true = [0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 2, 2, 2, 0, 1, 1, 2, 2]
    p = metrics.MultiClassPrecision(4, 'macro')
    p.update_state(y_true=y_true, y_pred=y_pred)
    np.testing.assert_allclose(p.result(),
                               sklearn.metrics.precision_score(y_true=y_true,
                                                               y_pred=y_pred, average='macro'),
                               rtol=1e-3)

    p = metrics.MultiClassPrecision(4, 'macro', top_k=2)
    p.update_state(y_true=y_true, y_pred=y_pred)
    np.testing.assert_allclose(p.result(), 7 / 12, rtol=1e-3)

    p.reset_states()
    assert p.result() == 0.0


def test_precision_micro():
    y_true = [0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 2, 2, 2, 0, 1, 1, 2, 2]
    p = metrics.MultiClassPrecision(4, 'micro')
    p.update_state(y_true=y_true, y_pred=y_pred)
    np.testing.assert_allclose(p.result(),
                               sklearn.metrics.precision_score(y_true=y_true,
                                                               y_pred=y_pred, average='micro'),
                               rtol=1e-3)

    p.reset_states()
    assert p.result() == 0.0

    p = metrics.MultiClassPrecision(4, 'micro', top_k=2)
    p.update_state(y_true=y_true, y_pred=y_pred)
    np.testing.assert_allclose(p.result(), 4/7, rtol=1e-3)

    p.reset_states()
    assert p.result() == 0.0


def test_recall_macro():
    y_true = [0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 2, 2, 2, 0, 1, 1, 2, 2]
    r = metrics.MultiClassRecall(4, 'macro')
    r.update_state(y_true, y_pred)
    np.testing.assert_allclose(r.result(),
                               sklearn.metrics.recall_score(y_true=y_true,
                                                            y_pred=y_pred, average='macro'),
                               rtol=1e-3)

    r.reset_states()
    assert r.result() == 0.0

    r = metrics.MultiClassRecall(4, 'macro', top_k=2)
    r.update_state(y_true=[0, 0, 0, 0, 2, 2, 2, 1, 1, 1], y_pred=[0, 1, 0, 1, 2, 2, 2, 0, 1, 1])
    np.testing.assert_allclose(r.result(), 7 / 12, rtol=1e-3)

    r.reset_states()
    assert r.result() == 0.0


def test_recall_micro():
    y_true = [0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 2, 2, 2, 0, 1, 1, 2, 2]
    r = metrics.MultiClassRecall(4, 'micro')
    r.update_state(y_true, y_pred)
    np.testing.assert_allclose(r.result(),
                               sklearn.metrics.recall_score(y_true=y_true,
                                                            y_pred=y_pred, average='micro'),
                               rtol=1e-3)

    r.reset_states()
    assert r.result() == 0.0

    r = metrics.MultiClassRecall(4, 'micro', top_k=2)
    r.update_state(y_true=y_true, y_pred=y_pred)
    np.testing.assert_allclose(r.result(), 0.442857, rtol=1e-3)

    r.reset_states()
    assert r.result() == 0.0


def test_f1_score_macro():
    y_true = [0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 2, 2, 2, 0, 1, 1, 2, 2]
    f1 = metrics.MultiClassF1Score(4, 'macro')
    f1.update_state(y_true, y_pred)
    np.testing.assert_allclose(f1.result(),
                               sklearn.metrics.f1_score(y_true=y_true,
                                                        y_pred=y_pred, average='macro'),
                               rtol=1e-3)

    f1.reset_states()
    assert f1.result() == 0.0

    f1 = metrics.MultiClassF1Score(4, 'macro', top_k=2)
    f1.update_state(y_true=y_true, y_pred=y_pred)
    np.testing.assert_allclose(f1.result(), 1/2, rtol=1e-3)

    f1.reset_states()
    assert f1.result() == 0.0


def test_f1_score_micro():
    y_true = [0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 2, 2, 2, 0, 1, 1, 2, 2]

    f1 = metrics.MultiClassF1Score(4, 'micro')
    f1.update_state(y_true, y_pred)
    np.testing.assert_allclose(f1.result(),
                               sklearn.metrics.f1_score(y_true=y_true,
                                                        y_pred=y_pred, average='micro'),
                               rtol=1e-3)

    f1.reset_states()
    assert f1.result() == 0.0

    f1 = metrics.MultiClassF1Score(4, 'micro', top_k=2)
    f1.update_state(y_true=y_true, y_pred=y_pred)
    np.testing.assert_allclose(f1.result(), 0.442857, rtol=1e-3)

    f1.reset_states()
    assert f1.result() == 0.0
