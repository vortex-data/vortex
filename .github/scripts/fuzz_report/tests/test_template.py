"""Tests for template module.

Critically verifies that rendered templates match the downstream grep patterns
used by fuzzer-fix-automation.yml to extract crash details from issue bodies.
"""

import os
import re
from pathlib import Path

import pytest

from fuzz_report.template import render_template, render_template_to_file


@pytest.fixture
def simple_template(tmp_path):
    """Create a simple template file."""
    content = "# {{TITLE}}\n\nTarget: {{TARGET}}\nValue: {{VALUE}}\nMissing: {{MISSING}}\n"
    path = tmp_path / "simple.md"
    path.write_text(content)
    return str(path)


@pytest.fixture
def conditional_template(tmp_path):
    """Create a template with Jinja2 conditionals."""
    content = "# Title\n{% if ANALYSIS %}\n### Analysis\n{{ANALYSIS}}\n{% endif %}\nFooter\n"
    path = tmp_path / "conditional.md"
    path.write_text(content)
    return str(path)


@pytest.fixture
def new_issue_template():
    """Path to the actual new_issue.md template."""
    return str(Path(__file__).parent.parent / "templates" / "new_issue.md")


@pytest.fixture
def related_comment_template():
    """Path to the actual related_comment.md template."""
    return str(Path(__file__).parent.parent / "templates" / "related_comment.md")


class TestRenderTemplate:
    def test_with_variables(self, simple_template):
        variables = {
            "TITLE": "Test Title",
            "TARGET": "file_io",
            "VALUE": "42",
        }
        result = render_template(simple_template, variables, use_env=False)
        assert "Test Title" in result
        assert "file_io" in result
        assert "42" in result
        assert "(not set)" in result  # MISSING

    def test_with_env_variables(self, simple_template):
        os.environ["TITLE"] = "Env Title"
        os.environ["TARGET"] = "env_target"
        os.environ["VALUE"] = "99"
        try:
            result = render_template(simple_template, use_env=True)
            assert "Env Title" in result
            assert "env_target" in result
            assert "99" in result
        finally:
            del os.environ["TITLE"]
            del os.environ["TARGET"]
            del os.environ["VALUE"]

    def test_variables_override_env(self, simple_template):
        os.environ["TITLE"] = "Env Title"
        try:
            result = render_template(
                simple_template,
                variables={"TITLE": "Override Title"},
                use_env=True,
            )
            assert "Override Title" in result
            assert "Env Title" not in result
        finally:
            del os.environ["TITLE"]

    def test_missing_variable(self, simple_template):
        result = render_template(simple_template, {}, use_env=False)
        assert result.count("(not set)") == 4


class TestConditionals:
    def test_if_shown_when_truthy(self, conditional_template):
        result = render_template(
            conditional_template,
            {"ANALYSIS": "Root cause is X"},
            use_env=False,
        )
        assert "### Analysis" in result
        assert "Root cause is X" in result

    def test_if_hidden_when_empty(self, conditional_template):
        result = render_template(
            conditional_template,
            {"ANALYSIS": ""},
            use_env=False,
        )
        assert "### Analysis" not in result

    def test_if_hidden_when_not_set(self, conditional_template):
        result = render_template(
            conditional_template,
            {},
            use_env=False,
        )
        assert "### Analysis" not in result

    def test_footer_always_present(self, conditional_template):
        result = render_template(conditional_template, {}, use_env=False)
        assert "Footer" in result


class TestRenderTemplateToFile:
    def test_writes_to_file(self, simple_template, tmp_path):
        variables = {"TITLE": "Test", "TARGET": "test", "VALUE": "1"}
        output_path = tmp_path / "output.md"
        render_template_to_file(simple_template, str(output_path), variables, use_env=False)
        content = output_path.read_text()
        assert "Test" in content


class TestDownstreamGrepCompatibility:
    """Verify that rendered templates match the grep patterns used by
    fuzzer-fix-automation.yml (lines 81-89) to extract crash details.

    These patterns are:
      TARGET=$(grep -oP '(?<=\\*\\*Target\\*\\*: `)[^`]+' issue_body.txt)
      CRASH_FILE=$(grep -oP '(?<=\\*\\*Crash File\\*\\*: `)[^`]+' issue_body.txt)
      ARTIFACT_URL=$(grep -oP 'https://[^\\s]+/artifacts/[0-9]+' issue_body.txt | head -1)
    """

    SAMPLE_VARS = {
        "FUZZ_TARGET": "file_io",
        "CRASH_FILE": "crash-abc123",
        "BRANCH": "develop",
        "COMMIT": "abc1234",
        "ARTIFACT_URL": "https://github.com/spiraldb/vortex/actions/runs/12345/artifacts/67890",
        "PANIC_MESSAGE": "index out of bounds: the len is 10 but the index is 15",
        "CRASH_LOCATION": "vortex-array/src/compute/slice.rs:142",
        "STACK_TRACE_RAW": "   0: std::panicking\n   1: vortex_array::compute::slice",
        "DEBUG_OUTPUT": "Array { dtype: Int32, len: 10 }",
        "SEED_HASH": "aaa111",
        "STACK_TRACE_HASH": "bbb222",
        "MESSAGE_HASH": "ccc333",
        "CLAUDE_ANALYSIS": "The crash is caused by an off-by-one error.",
    }

    def test_target_grep_pattern(self, new_issue_template):
        """Verify **Target**: `value` matches downstream grep."""
        rendered = render_template(new_issue_template, self.SAMPLE_VARS, use_env=False)
        # Simulate: grep -oP '(?<=\*\*Target\*\*: `)[^`]+'
        match = re.search(r"\*\*Target\*\*: `([^`]+)`", rendered)
        assert match is not None, f"Target pattern not found in:\n{rendered}"
        assert match.group(1) == "file_io"

    def test_crash_file_grep_pattern(self, new_issue_template):
        """Verify **Crash File**: `value` matches downstream grep."""
        rendered = render_template(new_issue_template, self.SAMPLE_VARS, use_env=False)
        # Simulate: grep -oP '(?<=\*\*Crash File\*\*: `)[^`]+'
        match = re.search(r"\*\*Crash File\*\*: `([^`]+)`", rendered)
        assert match is not None, f"Crash File pattern not found in:\n{rendered}"
        assert match.group(1) == "crash-abc123"

    def test_artifact_url_grep_pattern(self, new_issue_template):
        """Verify artifact URL matches downstream grep."""
        rendered = render_template(new_issue_template, self.SAMPLE_VARS, use_env=False)
        # Simulate: grep -oP 'https://[^\s]+/artifacts/[0-9]+'
        match = re.search(r"https://[^\s]+/artifacts/[0-9]+", rendered)
        assert match is not None, f"Artifact URL pattern not found in:\n{rendered}"
        assert "67890" in match.group(0)

    def test_hidden_hashes_present(self, new_issue_template):
        """Verify hidden HTML comment with hashes is present."""
        rendered = render_template(new_issue_template, self.SAMPLE_VARS, use_env=False)
        assert "<!-- seed_hash:aaa111" in rendered
        assert "stack_hash:bbb222" in rendered
        assert "message_hash:ccc333" in rendered

    def test_claude_analysis_shown_when_present(self, new_issue_template):
        """Claude analysis section should appear when provided."""
        rendered = render_template(new_issue_template, self.SAMPLE_VARS, use_env=False)
        assert "Root Cause Analysis" in rendered
        assert "off-by-one error" in rendered

    def test_claude_analysis_hidden_when_empty(self, new_issue_template):
        """Claude analysis section should not appear when empty."""
        vars_no_analysis = {**self.SAMPLE_VARS, "CLAUDE_ANALYSIS": ""}
        rendered = render_template(new_issue_template, vars_no_analysis, use_env=False)
        assert "Root Cause Analysis" not in rendered

    def test_stack_trace_in_details(self, new_issue_template):
        """Stack trace should be inside a <details> block."""
        rendered = render_template(new_issue_template, self.SAMPLE_VARS, use_env=False)
        assert "<details>" in rendered
        assert "Stack Trace" in rendered
        assert "Debug Output" not in rendered

    def test_related_comment_target_pattern(self, related_comment_template):
        """Related comment template should also have compatible Target pattern."""
        vars_with_dedup = {
            **self.SAMPLE_VARS,
            "DEDUP_REASON": "Same panic location",
            "DEDUP_CONFIDENCE": "high",
        }
        rendered = render_template(related_comment_template, vars_with_dedup, use_env=False)
        match = re.search(r"\*\*Target\*\*: `([^`]+)`", rendered)
        assert match is not None
        assert match.group(1) == "file_io"

    def test_full_end_to_end_grep_simulation(self, new_issue_template):
        """Full simulation of the three grep commands from fuzzer-fix-automation.yml."""
        rendered = render_template(new_issue_template, self.SAMPLE_VARS, use_env=False)

        # grep -oP '(?<=\*\*Target\*\*: `)[^`]+'
        target_match = re.search(r"(?<=\*\*Target\*\*: `)[^`]+", rendered)
        assert target_match is not None
        assert target_match.group(0) == "file_io"

        # grep -oP '(?<=\*\*Crash File\*\*: `)[^`]+'
        crash_match = re.search(r"(?<=\*\*Crash File\*\*: `)[^`]+", rendered)
        assert crash_match is not None
        assert crash_match.group(0) == "crash-abc123"

        # grep -oP 'https://[^\s]+/artifacts/[0-9]+'
        url_match = re.search(r"https://[^\s]+/artifacts/[0-9]+", rendered)
        assert url_match is not None
        assert (
            url_match.group(0)
            == "https://github.com/spiraldb/vortex/actions/runs/12345/artifacts/67890"
        )
