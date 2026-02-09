"""Template rendering for fuzzer crash reports.

Supports {{VAR}} substitution and {{#if VAR}}...{{/if}} conditionals.
"""

import os
import re
from pathlib import Path


def render_template(
    template_path: str | Path,
    variables: dict[str, str] | None = None,
    use_env: bool = True,
) -> str:
    """
    Render a template by substituting {{VAR}} placeholders and
    evaluating {{#if VAR}}...{{/if}} conditionals.

    Conditionals are shown if the variable is non-empty and not "(not set)".

    Args:
        template_path: Path to the template file
        variables: Dictionary of variables to substitute
        use_env: If True, also look up variables from environment

    Returns:
        Rendered template content
    """
    template_content = Path(template_path).read_text()
    variables = variables or {}

    def resolve_var(var_name: str) -> str:
        """Resolve a variable name to its value."""
        if var_name in variables:
            return str(variables[var_name])
        if use_env and var_name in os.environ:
            return os.environ[var_name]
        return "(not set)"

    def is_truthy(var_name: str) -> bool:
        """Check if a variable is considered truthy for conditionals."""
        value = resolve_var(var_name)
        return bool(value) and value != "(not set)"

    # Process {{#if VAR}}...{{/if}} conditionals (can be nested)
    def process_conditionals(content: str) -> str:
        # Process from innermost to outermost
        pattern = r"\{\{#if\s+([A-Z_][A-Z0-9_]*)\}\}(.*?)\{\{/if\}\}"
        while re.search(pattern, content, re.DOTALL):
            content = re.sub(
                pattern,
                lambda m: m.group(2) if is_truthy(m.group(1)) else "",
                content,
                flags=re.DOTALL,
            )
        return content

    content = process_conditionals(template_content)

    # Replace all {{VAR}} patterns
    var_pattern = r"\{\{([A-Z_][A-Z0-9_]*)\}\}"
    content = re.sub(var_pattern, lambda m: resolve_var(m.group(1)), content)

    # Clean up any double blank lines left by removed conditionals
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content


def render_template_to_file(
    template_path: str | Path,
    output_path: str | Path,
    variables: dict[str, str] | None = None,
    use_env: bool = True,
) -> None:
    """Render template and write to output file."""
    content = render_template(template_path, variables, use_env)
    Path(output_path).write_text(content)
