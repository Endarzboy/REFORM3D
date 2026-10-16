"""Reference similarities use known vectors and preserve the native 3D order."""
import numpy as np
import pytest

from reform3d.feature_similarity import reference_similarity


def test_equal_opposite_orthogonal_vectors_keep_c_order_grid():
    tokens = np.array([
        [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [3., 0., 0.], [1., 1., 0.], [0., -1., 0.], [-1., -1., 0.],
    ])
    result = reference_similarity(tokens, (2, 2, 2), (1, 0, 0))
    expected = np.array([1., -1., 0., 0., 1., 2 ** -.5, 0., -2 ** -.5])
    assert result.shape == (2, 2, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result.ravel(), expected, atol=1e-7)
    assert result[1, 0, 0] == 1.


def test_nonfirst_reference_matches_independent_cosine_and_preserves_input():
    tokens = np.random.default_rng(13).normal(size=(24, 11)).astype(np.float32)
    original = tokens.copy()
    tokens.setflags(write=False)
    result = reference_similarity(tokens, (2, 3, 4), (1, 1, 2))
    reference_index = 1 * 3 * 4 + 1 * 4 + 2
    norms = np.linalg.norm(tokens.astype(np.float64), axis=1)
    expected = (tokens.astype(np.float64) @ tokens[reference_index]) / (
        norms * norms[reference_index])
    np.testing.assert_allclose(result.ravel(), expected, atol=1e-7)
    np.testing.assert_array_equal(tokens, original)
    assert result[1, 1, 2] == 1.
    assert np.isfinite(result).all()
    assert np.all((result >= -1) & (result <= 1))


def test_positive_per_patch_scaling_does_not_change_similarity():
    tokens = np.array([[1., -2., 3.], [-1., 5., 2.], [3., 2., -1.], [-2., -1., -4.]])
    expected = reference_similarity(tokens, (1, 2, 2), (0, 1, 0))
    actual = reference_similarity(tokens * np.array([1e200, 1e-200, 3., .1])[:, None],
                                  (1, 2, 2), (0, 1, 0))
    np.testing.assert_array_equal(actual, expected)


def test_finite_huge_and_subnormal_features_have_stable_norms():
    maximum = np.finfo(np.float64).max
    minimum = np.nextafter(0., 1.)
    tokens = np.array([[maximum, maximum], [-maximum, -maximum], [minimum, minimum]])
    result = reference_similarity(tokens, (1, 1, 3), (0, 0, 0))
    np.testing.assert_array_equal(result, [[[1., -1., 1.]]])


def test_integer_features_numpy_indices_and_single_patch_are_supported():
    result = reference_similarity([[3, 4]], np.array([1, 1, 1]),
                                  (np.int64(0), np.int32(0), np.int64(0)))
    np.testing.assert_array_equal(result, np.ones((1, 1, 1), np.float32))


@pytest.mark.parametrize('tokens', [
    np.ones(4), np.ones((1, 2, 2)), np.ones((0, 2)), np.ones((4, 0)),
    np.ones((3, 2)), np.zeros((4, 2)), [[1., 2.], [0., 0.], [1., 2.], [1., 2.]],
    np.full((4, 2), np.nan), np.full((4, 2), np.inf),
    np.full((4, 2), -np.inf), np.full((4, 2), 1 + 0j),
    np.full((4, 2), '1'), np.full((4, 2), True),
    np.array([[1, 2]] * 4, dtype=object), [[1], [1, 2]], None,
])
def test_invalid_features_fail_explicitly(tokens):
    with pytest.raises(ValueError, match='tokens|feature|norm'):
        reference_similarity(tokens, (1, 2, 2), (0, 0, 0))


@pytest.mark.parametrize('shape', [
    (), (1, 4), (1, 1, 2, 2), (0, 2, 2), (-1, 2, 2),
    (1., 2, 2), (True, 2, 2), (1, np.bool_(True), 4), '122', None, 4,
])
def test_invalid_grid_shape_fails_explicitly(shape):
    with pytest.raises(ValueError, match='grid_shape'):
        reference_similarity(np.ones((4, 2)), shape, (0, 0, 0))


@pytest.mark.parametrize('reference', [
    (), (0, 0), (0, 0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, 2, 0),
    (0, 0, 2), (0., 0, 0), (True, 0, 0), (np.bool_(False), 0, 0),
    '000', None, 0,
])
def test_invalid_reference_patch_fails_explicitly(reference):
    with pytest.raises(ValueError, match='reference_patch'):
        reference_similarity(np.ones((4, 2)), (1, 2, 2), reference)
