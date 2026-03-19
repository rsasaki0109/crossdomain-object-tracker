"""Tests for CLI entry point and subcommand help."""

from __future__ import annotations

import pytest

from crossdomain_object_tracker.cli import main


class TestCLIHelp:
    """Verify that CLI --help for each subcommand exits cleanly."""

    def test_cli_help(self) -> None:
        """Running main with --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_download_help(self) -> None:
        """Running download --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["download", "--help"])
        assert exc_info.value.code == 0

    def test_cli_detect_help(self) -> None:
        """Running detect --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["detect", "--help"])
        assert exc_info.value.code == 0

    def test_cli_latex_help(self) -> None:
        """Running latex --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["latex", "--help"])
        assert exc_info.value.code == 0

    def test_cli_evaluate_help(self) -> None:
        """Running evaluate --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["evaluate", "--help"])
        assert exc_info.value.code == 0

    def test_cli_visualize_help(self) -> None:
        """Running visualize --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["visualize", "--help"])
        assert exc_info.value.code == 0

    def test_cli_report_help(self) -> None:
        """Running report --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["report", "--help"])
        assert exc_info.value.code == 0

    def test_cli_no_command(self) -> None:
        """Running main with no arguments exits with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1
