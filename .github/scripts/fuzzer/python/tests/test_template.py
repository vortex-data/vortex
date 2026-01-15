"""Tests for template module."""

import os
import tempfile
from pathlib import Path

import pytest

from ..template import render_template, render_template_to_file


@pytest.fixture
def template_file():
    """Create a temporary template file."""
    content = """# {{TITLE}}

Target: {{TARGET}}
Value: {{VALUE}}
Missing: {{MISSING}}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    Path(f.name).unlink()


class TestRenderTemplate:
    def test_with_variables(self, template_file):
        variables = {
            "TITLE": "Test Title",
            "TARGET": "file_io",
            "VALUE": "42",
        }
        result = render_template(template_file, variables, use_env=False)

        assert "Test Title" in result
        assert "file_io" in result
        assert "42" in result
        assert "(not set)" in result  # MISSING

    def test_with_env_variables(self, template_file):
        os.environ["TITLE"] = "Env Title"
        os.environ["TARGET"] = "env_target"
        os.environ["VALUE"] = "99"

        try:
            result = render_template(template_file, use_env=True)
            assert "Env Title" in result
            assert "env_target" in result
            assert "99" in result
        finally:
            del os.environ["TITLE"]
            del os.environ["TARGET"]
            del os.environ["VALUE"]

    def test_variables_override_env(self, template_file):
        os.environ["TITLE"] = "Env Title"

        try:
            result = render_template(
                template_file,
                variables={"TITLE": "Override Title"},
                use_env=True,
            )
            assert "Override Title" in result
            assert "Env Title" not in result
        finally:
            del os.environ["TITLE"]

    def test_missing_variable(self, template_file):
        result = render_template(template_file, {}, use_env=False)
        # All variables should be "(not set)"
        assert result.count("(not set)") == 4


class TestRenderTemplateToFile:
    def test_writes_to_file(self, template_file):
        variables = {"TITLE": "Test", "TARGET": "test", "VALUE": "1"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            output_path = f.name

        try:
            render_template_to_file(template_file, output_path, variables, use_env=False)
            content = Path(output_path).read_text()
            assert "Test" in content
        finally:
            Path(output_path).unlink()
