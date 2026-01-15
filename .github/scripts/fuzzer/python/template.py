"""Template rendering for fuzzer crash reports."""

import os
import re
from pathlib import Path


def render_template(
    template_path: str | Path,
    variables: dict[str, str] | None = None,
    use_env: bool = True,
) -> str:
    """
    Render a template by substituting {{VAR}} placeholders.

    Args:
        template_path: Path to the template file
        variables: Dictionary of variables to substitute
        use_env: If True, also look up variables from environment

    Returns:
        Rendered template content
    """
    template_content = Path(template_path).read_text()
    variables = variables or {}

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)

        # Try provided variables first
        if var_name in variables:
            return str(variables[var_name])

        # Try environment variables
        if use_env and var_name in os.environ:
            return os.environ[var_name]

        return "(not set)"

    # Replace all {{VAR}} patterns
    pattern = r"\{\{([A-Z_][A-Z0-9_]*)\}\}"
    return re.sub(pattern, replace_var, template_content)


def render_template_to_file(
    template_path: str | Path,
    output_path: str | Path,
    variables: dict[str, str] | None = None,
    use_env: bool = True,
) -> None:
    """Render template and write to output file."""
    content = render_template(template_path, variables, use_env)
    Path(output_path).write_text(content)
