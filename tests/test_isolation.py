import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import anthropic
import pytest
from anthropic.lib.credentials import (
    CredentialsFile,
    StaticToken,
    WorkloadIdentityCredentials,
    default_credentials,
)
from pytest_socket import SocketBlockedError

REPO = Path(__file__).resolve().parent.parent


def _plant_profile(config_dir: Path, profile: str) -> None:
    (config_dir / "configs").mkdir(parents=True, exist_ok=True)
    (config_dir / "credentials").mkdir(parents=True, exist_ok=True)
    config = config_dir / "configs" / f"{profile}.json"
    config.write_text(json.dumps({"authentication": {"type": "user_oauth"}}))
    config.chmod(0o644)
    creds = config_dir / "credentials" / f"{profile}.json"
    creds.write_text(json.dumps({"access_token": "fake-token", "expires_at": 4102444800}))
    creds.chmod(0o600)


def _plant_signals(tmp_path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    """Ways a developer machine can be logged in to the API, keyed by variant."""
    home = tmp_path / "home"
    _plant_profile(home / ".config" / "anthropic", "default")
    _plant_profile(home / ".config" / "anthropic", "dev")
    _plant_profile(tmp_path / "elsewhere", "default")
    return home, {
        "disk_profile_in_home": {},
        "api_key": {"ANTHROPIC_API_KEY": "sk-fake"},
        "auth_token": {"ANTHROPIC_AUTH_TOKEN": "fake"},
        "named_profile": {"ANTHROPIC_PROFILE": "dev"},
        "config_dir": {"ANTHROPIC_CONFIG_DIR": str(tmp_path / "elsewhere")},
        "workload_identity": {
            "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_fake",
            "ANTHROPIC_ORGANIZATION_ID": "org_fake",
            "ANTHROPIC_IDENTITY_TOKEN": "fake.jwt.token",
        },
    }


@pytest.mark.parametrize(
    "variant",
    [
        "disk_profile_in_home",
        "api_key",
        "auth_token",
        "named_profile",
        "config_dir",
        "workload_identity",
    ],
)
def test_planted_signal_is_found_by_sdk(monkeypatch, tmp_path, variant):
    # Control for the scrub test below: each planted signal really authenticates.
    home, signals = _plant_signals(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    for name, value in signals[variant].items():
        monkeypatch.setenv(name, value)

    if variant == "api_key":
        assert anthropic.Anthropic().api_key == "sk-fake"
        return
    provider = default_credentials().provider
    if variant == "auth_token":
        assert isinstance(provider, StaticToken)
    elif variant == "workload_identity":
        assert isinstance(provider, WorkloadIdentityCredentials)
    else:
        assert isinstance(provider, CredentialsFile)
        expected_dir = tmp_path / "elsewhere" if variant == "config_dir" else home
        assert provider.config_path.is_relative_to(expected_dir)
        assert provider.profile == ("dev" if variant == "named_profile" else "default")


SCRUB_INNER_TEST = """
import anthropic
import pytest
from anthropic.lib.credentials import default_credentials


def test_sdk_finds_no_credentials():
    assert default_credentials() is None
    client = anthropic.Anthropic(max_retries=0)
    assert client.api_key is None and client.auth_token is None
    with pytest.raises(TypeError, match="Could not resolve authentication method"):
        client.messages.create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}]
        )
"""


def run_scrub_check(tmp_path: Path, conftest: str) -> subprocess.CompletedProcess:
    """Run a pytest session with every planted signal at once under the given conftest."""
    home, signals = _plant_signals(tmp_path)
    # Anything that slips through goes to a closed local port, not to the API.
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
    }
    for values in signals.values():
        env.update(values)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "conftest.py").write_text(conftest)
    (proj / "test_inner.py").write_text(SCRUB_INNER_TEST)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--disable-socket"],
        cwd=proj,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_conftest_hides_all_credentials_from_sdk(tmp_path):
    result = run_scrub_check(tmp_path, (REPO / "tests" / "conftest.py").read_text())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.mark.filterwarnings("ignore:A test tried to use socket:UserWarning")
def test_unix_sockets_are_blocked():
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)


IMPORT_TIME_INNER_TEST = """
import os
import socket

import pytest
from pytest_socket import SocketBlockedError

SECRETS_AT_IMPORT = sorted(n for n in os.environ if n.startswith("ANTHROPIC_"))
HOME_AT_IMPORT = os.environ["HOME"]
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
    SOCKET_AT_IMPORT = "open"
except SocketBlockedError:
    SOCKET_AT_IMPORT = "blocked"
try:
    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()
    UNIX_AT_IMPORT = "open"
except SocketBlockedError:
    UNIX_AT_IMPORT = "blocked"


def test_environment_was_scrubbed_before_collection():
    assert SECRETS_AT_IMPORT == []
    assert not os.path.exists(os.path.join(HOME_AT_IMPORT, "real-home-marker"))
    assert SOCKET_AT_IMPORT == "blocked"
    assert UNIX_AT_IMPORT == "blocked"


def test_tcp_blocked():
    with pytest.raises(SocketBlockedError):
        socket.create_connection(("127.0.0.1", 9), timeout=1)


def test_unix_blocked():
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
"""


def test_isolation_applies_from_collection_without_pytest_config(tmp_path):
    # -c /dev/null: no pyproject addopts, so the conftest alone must do the blocking.
    home = tmp_path / "home"
    home.mkdir()
    (home / "real-home-marker").touch()
    proj = tmp_path / "proj"
    proj.mkdir()
    shutil.copy(REPO / "tests" / "conftest.py", proj / "conftest.py")
    (proj / "test_inner.py").write_text(IMPORT_TIME_INNER_TEST)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", os.devnull],
        cwd=proj,
        env={"PATH": os.environ["PATH"], "HOME": str(home), "ANTHROPIC_API_KEY": "sk-fake"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout


def test_gitignore_covers_local_dir_as_symlink(tmp_path):
    # Isolated repo: the checkout's own info/exclude must not mask a broken pattern.
    shutil.copy(REPO / ".gitignore", tmp_path / ".gitignore")
    git = ["git", "-c", "core.excludesFile=/dev/null", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    (tmp_path / "_local").symlink_to(tmp_path / "elsewhere")
    result = subprocess.run([*git, "check-ignore", "--no-index", "-q", "_local"])
    assert result.returncode == 0


def test_qa_refuses_to_run_without_venv(tmp_path):
    shutil.copy(REPO / "qa.sh", tmp_path / "qa.sh")
    # A foreign environment with the tools on PATH must not be used instead.
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(
        ["bash", str(tmp_path / "qa.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
    assert ".venv missing" in result.stderr
    assert result.stdout == ""


PER_TEST_INNER = """
import os

import pytest

SESSION = {}


@pytest.fixture(scope="session")
def session_env():
    return {
        "secrets": sorted(n for n in os.environ if n.startswith("ANTHROPIC_")),
        "home": os.environ["HOME"],
    }


def test_a_session_fixture_sees_the_scrubbed_environment(session_env):
    assert session_env["secrets"] == []
    assert session_env["home"] != SESSION_REAL_HOME
    SESSION["home"] = os.environ["HOME"]


def test_b_leaks_a_credential_without_monkeypatch():
    os.environ["ANTHROPIC_API_KEY"] = "sk-leaked"


def test_c_does_not_see_the_leak_and_has_its_own_home():
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert os.environ["HOME"] != SESSION["home"]
"""


def test_per_test_fixture_scrubs_credentials_and_gives_each_test_a_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    proj = tmp_path / "proj"
    proj.mkdir()
    shutil.copy(REPO / "tests" / "conftest.py", proj / "conftest.py")
    inner = PER_TEST_INNER.replace("SESSION_REAL_HOME", repr(str(home)))
    (proj / "test_inner.py").write_text(inner)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=proj,
        env={"PATH": os.environ["PATH"], "HOME": str(home), "ANTHROPIC_API_KEY": "sk-fake"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout


def test_qa_uses_the_tools_of_its_own_venv(tmp_path):
    shutil.copy(REPO / "qa.sh", tmp_path / "qa.sh")
    venv_bin = tmp_path / ".venv" / "bin"
    foreign = tmp_path / "foreign"
    venv_bin.mkdir(parents=True)
    foreign.mkdir()
    log = tmp_path / "calls.log"
    for directory, name, tag in (
        (venv_bin, "ruff", "venv-ruff"),
        (venv_bin, "python", "venv-python"),
        (foreign, "ruff", "foreign-ruff"),
        (foreign, "pytest", "foreign-pytest"),
    ):
        script = directory / name
        script.write_text(f'#!/bin/sh\necho "{tag} $*" >> "{log}"\n')
        script.chmod(0o755)
    env = {"PATH": f"{foreign}:/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(
        ["bash", str(tmp_path / "qa.sh")], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        "venv-ruff check .",
        "venv-ruff format --check .",
        "venv-python -m pytest -q",
    ]


def test_unconfigure_restores_the_socket_class_found_at_start():
    script = f"""
import socket
import sys

sys.path.insert(0, {str(REPO / "tests")!r})
import conftest


class Found(socket.socket):
    pass


socket.socket = Found
conftest.pytest_configure(None)
assert socket.socket is not Found
conftest.pytest_unconfigure(None)
print(socket.socket.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Found"
