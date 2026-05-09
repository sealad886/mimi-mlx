from __future__ import annotations

import mlx.core as mx
import numpy as np

from mimi_mlx.quantizer import (
    EuclideanCodebook,
    ResidualVectorQuantization,
    SplitResidualVectorQuantizer,
)


def test_euclidean_codebook_chooses_nearest_centroid():
    codebook = EuclideanCodebook(mx.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]))
    vectors = mx.array([[[0.1, 0.2], [1.8, 0.1], [0.1, 1.9]]])

    codes = codebook.encode(vectors)

    assert np.array_equal(np.array(codes), np.array([[0, 1, 2]]))
    decoded = codebook.decode(codes)
    assert np.array_equal(np.array(decoded), np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]]))


def test_residual_vector_quantization_uses_ordered_residuals():
    first = EuclideanCodebook(mx.array([[0.0], [2.0]]))
    second = EuclideanCodebook(mx.array([[0.0], [0.75]]))
    rvq = ResidualVectorQuantization([first, second])

    codes = rvq.encode(mx.array([[[2.6], [0.7]]]))
    decoded = rvq.decode(codes)

    assert np.array_equal(np.array(codes), np.array([[[1, 1], [0, 1]]]))
    assert np.allclose(np.array(decoded), np.array([[[2.75], [0.75]]]))


def test_split_quantizer_keeps_semantic_then_acoustic_order():
    semantic = ResidualVectorQuantization([EuclideanCodebook(mx.array([[0.0], [10.0]]))])
    acoustic = ResidualVectorQuantization(
        [
            EuclideanCodebook(mx.array([[0.0], [1.0]])),
            EuclideanCodebook(mx.array([[0.0], [0.25]])),
        ]
    )
    split = SplitResidualVectorQuantizer(semantic=semantic, acoustic=acoustic)

    codes = split.encode(mx.array([[[1.2], [9.8]]]))

    assert codes.shape == (1, 2, 3)
    assert np.array_equal(np.array(codes), np.array([[[0, 1, 1], [1, 1, 1]]]))
    assert int(mx.min(codes)) >= 0
    assert int(mx.max(codes)) < 2
