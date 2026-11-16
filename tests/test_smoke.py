import os
import socket

import pytest
from pytest_socket import SocketBlockedError

import llmeval


def test_package_imports_with_version():
    assert isinstance(llmeval.__version__, str)
    assert llmeval.__version__


@pytest.mark.filterwarnings("ignore:A test tried to use socket:UserWarning")
def test_network_is_blocked_in_tests():
    with pytest.raises(SocketBlockedError):
        socket.create_connection(("127.0.0.1", 9), timeout=1)


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
def test_api_credentials_are_not_in_environment(name):
    assert name not in os.environ


def test_project_is_mit_licensed():
    import tomllib
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    project = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    text = (repo / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License\n\nCopyright (c) 2026 Jan Tydlačka\n")
