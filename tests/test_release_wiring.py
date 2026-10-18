from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_pull_requests_run_tests_without_publishing_images():
    workflow = (ROOT / ".github/workflows/build.yml").read_text()
    assert "pull_request:" in workflow
    assert "inputs.publish_candidate == true" in workflow


def test_manual_candidate_publish_is_opt_in_and_never_uses_release_tags():
    workflow = (ROOT / ".github/workflows/build.yml").read_text()
    assert "publish_candidate:" in workflow
    assert "type: boolean" in workflow
    assert "default: false" in workflow
    assert 'runner_release="ghcr.io/singularity-layer/sgl-train:candidate-${GITHUB_SHA}"' in workflow
    assert 'seed_release="ghcr.io/singularity-layer/sgl-train:seed-candidate-${GITHUB_SHA}"' in workflow
    assert 'echo "ghcr.io/singularity-layer/sgl-train:${GITHUB_SHA}"' in workflow
    assert 'echo "ghcr.io/singularity-layer/sgl-train:seed-${GITHUB_SHA}"' in workflow
    assert 'if [[ "${{ github.event_name }}" == "push" ]]' in workflow


def test_package_image_and_seed_runtime_versions_stay_in_sync():
    package = (ROOT / "pyproject.toml").read_text()
    workflow = (ROOT / ".github/workflows/build.yml").read_text()
    seed = (ROOT / "scripts/seed-volume-runtime.sh").read_text()
    docker = (ROOT / "Dockerfile").read_text()
    seed_docker = (ROOT / "Dockerfile.seed").read_text()
    assert 'version = "0.7.0"' in package
    assert "sgl-train:v0.7.0" in workflow
    assert "sgl-train:seed-v0.7.0" in workflow
    assert "SGL_RUNTIME_VERSION:-v0.7.0" in seed
    assert "SGL_RUNTIME_VERSION=v0.7.0" in docker
    assert "SGL_RUNTIME_ROOT=/workspace/sgl-train/runtime/v0.7.0" in docker
    assert "SGL_RUNTIME_VERSION=v0.7.0" in seed_docker
    assert "decision-runtime-constraints.txt" in seed_docker
    assert "decision-venv/lib/python3.10/site-packages" in seed
