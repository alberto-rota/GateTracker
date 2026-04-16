import pytest
import torch

from gatetracker.geometry.projections import BackProject, Project, Warp
from gatetracker.geometry.transforms import euler2mat, mat2euler, Tdist


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def image_dims():
    return (384, 384)


def test_backproject_creates_point_cloud(device, batch_size, image_dims):
    H, W = image_dims
    backproject = BackProject(H, W).to(device)
    depth = torch.ones(batch_size, 1, H, W, device=device)
    K_inv = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    cloud = backproject(depth, K_inv)
    assert cloud.shape == (batch_size, 4, H * W)


def test_project_reprojects_cloud(device, batch_size, image_dims):
    H, W = image_dims
    project = Project(H, W).to(device)
    cloud = torch.randn(batch_size, 4, H * W, device=device)
    cloud[:, 2, :] = cloud[:, 2, :].abs() + 0.1
    cloud[:, 3, :] = 1.0
    K = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    T = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    coords, depth = project(cloud, K, T)
    assert coords.shape == (batch_size, H, W, 2)
    assert depth.shape == (batch_size, 1, H, W)


def test_euler2mat_identity():
    angles = torch.zeros(1, 3)
    R = euler2mat(angles)
    assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-5)


def test_mat2euler_roundtrip():
    angles = torch.tensor([[0.1, 0.2, 0.3]])
    R = euler2mat(angles)
    recovered = mat2euler(R)
    assert torch.allclose(angles, recovered, atol=1e-4)


def test_Tdist_zero_displacement():
    T1 = torch.eye(4).unsqueeze(0)
    T2 = torch.eye(4).unsqueeze(0)
    dist = Tdist(T1, T2)
    assert dist.item() < 1e-6
