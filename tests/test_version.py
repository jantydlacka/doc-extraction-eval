import os
import subprocess

from llmeval.report import tool_commit

GIT = [
    "git",
    "-c",
    "user.name=t",
    "-c",
    "user.email=t@example.invalid",
    "-c",
    "commit.gpgsign=false",
]


def _repo(tmp_path):
    subprocess.run([*GIT, "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("x = 1\n")
    # The file tool_commit looks for, tracked as in the tool's own checkout.
    (tmp_path / "src" / "llmeval").mkdir(parents=True)
    (tmp_path / "src" / "llmeval" / "report.py").write_text("x = 1\n")
    subprocess.run([*GIT, "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True)
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return head.stdout.strip()


def test_tool_commit_is_the_head_of_the_source_checkout(tmp_path):
    head = _repo(tmp_path)
    assert tool_commit(tmp_path) == head


def test_tool_commit_says_when_tracked_files_have_changed(tmp_path):
    head = _repo(tmp_path)
    (tmp_path / "pkg" / "module.py").write_text("x = 2\n")
    (tmp_path / "untracked.txt").write_text("results\n")
    assert tool_commit(tmp_path) == f"{head} (with uncommitted changes)"


def test_tool_commit_is_unknown_outside_a_git_checkout(tmp_path):
    assert tool_commit(tmp_path) == "unknown"


def test_tool_commit_is_unknown_without_git(tmp_path, monkeypatch):
    _repo(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "no-git-here"))
    assert tool_commit(tmp_path) == "unknown"


def test_tool_commit_ignores_a_git_dir_from_the_environment(tmp_path, monkeypatch):
    head = _repo(tmp_path / "tool")
    other = tmp_path / "other"
    _repo(other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    assert tool_commit(tmp_path / "tool") == head
    assert os.environ["GIT_DIR"]


def test_tool_commit_is_unknown_for_a_copy_inside_another_repository(tmp_path):
    _repo(tmp_path)
    copy = tmp_path / "vendor" / "tool"
    copy.mkdir(parents=True)
    (copy / "module.py").write_text("x = 1\n")
    assert tool_commit(copy) == "unknown"


def test_tool_commit_looks_at_the_repository_root_by_default():
    from llmeval import report

    assert tool_commit.__defaults__ == (report.PACKAGE_DIR.parents[1],)


def test_tool_commit_is_unknown_for_files_unpacked_at_the_root_of_another_repository(tmp_path):
    subprocess.run([*GIT, "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "other.txt").write_text("not the tool\n")
    subprocess.run([*GIT, "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(tmp_path), "commit", "-q", "-m", "other"], check=True)
    (tmp_path / "src" / "llmeval").mkdir(parents=True)
    (tmp_path / "src" / "llmeval" / "report.py").write_text("x = 1\n")  # untracked
    assert tool_commit(tmp_path) == "unknown"


def test_tool_commit_of_a_checkout_reached_through_a_symlink(tmp_path):
    head = _repo(tmp_path / "real")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert tool_commit(tmp_path / "link") == head


def test_tilde_replaces_only_the_home_directory_itself(monkeypatch):
    from llmeval.report import _tilde

    monkeypatch.setenv("HOME", "/home/user")
    assert _tilde("ledger /home/user/spend.jsonl") == "ledger ~/spend.jsonl"
    assert _tilde("ledger /home/userx/spend.jsonl") == "ledger /home/userx/spend.jsonl"
