"""The command-line entry point lists its commands and every command parses --help."""
import pytest

from skill2fsm import __version__
from skill2fsm.cli import COMMANDS, main


def test_usage_lists_every_command(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in COMMANDS)


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"skill2fsm {__version__}"


def test_unknown_command_is_rejected(capsys):
    assert main(["no-such-command"]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_command_help(name, capsys):
    with pytest.raises(SystemExit) as exc:
        main([name, "--help"])
    assert exc.value.code == 0
    assert f"skill2fsm {name}" in capsys.readouterr().out
