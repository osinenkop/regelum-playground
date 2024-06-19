import pytest
from src.critic import calculate_critic_targets
import torch


@pytest.fixture
def running_objectives():
    return torch.FloatTensor([[27.0], [23.0], [20.0]])


@pytest.fixture
def critic_model_output():
    return torch.FloatTensor([[1.0], [2.0], [3.0]])


@pytest.fixture
def times():
    return torch.FloatTensor([[0.0], [0.5], [1.0]])


def test_calculate_critic_targets_td_n_1(
    running_objectives: torch.FloatTensor,
    critic_model_output: torch.FloatTensor,
    times: torch.FloatTensor,
):
    discount_factor = 0.85

    r1, r2, r3 = (
        running_objectives[0, 0],
        running_objectives[1, 0],
        running_objectives[2, 0],
    )

    v1, v2, v3 = (
        critic_model_output[0, 0],
        critic_model_output[1, 0],
        critic_model_output[2, 0],
    )

    t1, t2, t3 = (times[0, 0], times[1, 0], times[2, 0])

    ct1 = r1 + discount_factor**t2 * v2
    ct2 = r2 + discount_factor**t2 * v3
    ct3 = r3

    critic_targets = calculate_critic_targets(
        running_objectives=running_objectives,
        critic_model_outputs=critic_model_output,
        times=times,
        td_n=1,
        discount_factor=discount_factor,
    )

    assert torch.allclose(torch.FloatTensor([[ct1], [ct2], [ct3]]), critic_targets)


def test_calculate_critic_targets_td_n_2(
    running_objectives: torch.FloatTensor,
    critic_model_output: torch.FloatTensor,
    times: torch.FloatTensor,
):
    discount_factor = 0.85
    r1, r2, r3 = (
        running_objectives[0, 0],
        running_objectives[1, 0],
        running_objectives[2, 0],
    )
    v1, v2, v3 = (
        critic_model_output[0, 0],
        critic_model_output[1, 0],
        critic_model_output[2, 0],
    )
    t1, t2, t3 = (
        times[0, 0],
        times[1, 0],
        times[2, 0],
    )

    ct1 = r1 + discount_factor**t2 * r2 + discount_factor**t3 * v3
    ct2 = r2 + discount_factor**t2 * r3
    ct3 = r3

    critic_targets = calculate_critic_targets(
        running_objectives=running_objectives,
        critic_model_outputs=critic_model_output,
        times=times,
        td_n=2,
        discount_factor=discount_factor,
    )

    assert torch.allclose(torch.FloatTensor([[ct1], [ct2], [ct3]]), critic_targets)


def test_calculate_critic_targets_big_td_n(
    running_objectives: torch.FloatTensor,
    critic_model_output: torch.FloatTensor,
    times: torch.FloatTensor,
):
    discount_factor = 0.85
    r1, r2, r3 = (
        running_objectives[0, 0],
        running_objectives[1, 0],
        running_objectives[2, 0],
    )
    v1, v2, v3 = (
        critic_model_output[0, 0],
        critic_model_output[1, 0],
        critic_model_output[2, 0],
    )
    t1, t2, t3 = (
        times[0, 0],
        times[1, 0],
        times[2, 0],
    )

    ct1 = r1 + discount_factor**t2 * r2 + discount_factor**t3 * r3
    ct2 = r2 + discount_factor**t2 * r3
    ct3 = r3

    critic_targets = calculate_critic_targets(
        running_objectives=running_objectives,
        critic_model_outputs=critic_model_output,
        times=times,
        td_n=3,
        discount_factor=discount_factor,
    )

    assert torch.allclose(torch.FloatTensor([[ct1], [ct2], [ct3]]), critic_targets)

    critic_targets_another_big_td_n = calculate_critic_targets(
        running_objectives=running_objectives,
        critic_model_outputs=critic_model_output,
        times=times,
        td_n=5,
        discount_factor=discount_factor,
    )

    assert torch.allclose(
        torch.FloatTensor([[ct1], [ct2], [ct3]]), critic_targets_another_big_td_n
    )
