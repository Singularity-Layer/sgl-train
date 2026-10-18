from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_pull_requests_run_tests_without_publishing_images():
    workflow = (ROOT / ".github/workflows/build.yml").read_text()
    assert "pull_request:" in workflow
    assert "if: github.event_name == 'push'" in workflow


def test_package_image_and_seed_runtime_versions_stay_in_sync():
    package = (ROOT / "pyproject.toml").read_text()
    workflow = (ROOT / ".github/workflows/build.yml").read_text()
    seed = (ROOT / "scripts/seed-volume-runtime.sh").read_text()
    docker = (ROOT / "Dockerfile").read_text()
    seed_docker = (ROOT / "Dockerfile.seed").read_text()
    assert 'version = "0.6.0"' in package
    assert "sgl-train:v0.6.0" in workflow
    assert "sgl-train:seed-v0.6.0" in workflow
    assert "SGL_RUNTIME_VERSION:-v0.6.0" in seed
    assert "SGL_RUNTIME_VERSION=v0.6.0" in docker
    assert "SGL_RUNTIME_ROOT=/workspace/sgl-train/runtime/v0.6.0" in docker
    assert "SGL_RUNTIME_VERSION=v0.6.0" in seed_docker
