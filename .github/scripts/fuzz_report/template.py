"""Template rendering for fuzzer crash reports using Jinja2."""

import os
from pathlib import Path

from jinja2 import BaseLoader, Environment, Undefined


class _NotSetUndefined(Undefined):
    """Jinja2 undefined that renders as '(not set)' and is falsy."""

    def __str__(self):
        return "(not set)"

    def __bool__(self):
        return False


def render_template(
    template_path: str | Path,
    variables: dict[str, str] | None = None,
    use_env: bool = True,
) -> str:
    """
    Render a Jinja2 template with the given variables.

    Variables are resolved in order: explicit variables > environment variables.
    Undefined variables render as "(not set)".

    Args:
        template_path: Path to the template file
        variables: Dictionary of variables to substitute
        use_env: If True, also look up variables from environment

    Returns:
        Rendered template content
    """
    template_content = Path(template_path).read_text()
    merged = {}
    if use_env:
        merged.update(os.environ)
    if variables:
        merged.update(variables)

    env = Environment(
        loader=BaseLoader(),
        keep_trailing_newline=True,
        undefined=_NotSetUndefined,
    )
    template = env.from_string(template_content)
    return template.render(merged)


def render_template_to_file(
    template_path: str | Path,
    output_path: str | Path,
    variables: dict[str, str] | None = None,
    use_env: bool = True,
) -> None:
    """Render template and write to output file."""
    content = render_template(template_path, variables, use_env)
    Path(output_path).write_text(content)
