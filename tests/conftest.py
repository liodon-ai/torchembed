import pytest


@pytest.fixture(params=[2, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[4, 8])
def num_heads(request):
    return request.param


@pytest.fixture(params=[8, 16])
def seq_len(request):
    return request.param


@pytest.fixture(params=[32, 64])
def dim(request):
    return request.param
