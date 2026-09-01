import sys

import click

from . import mcp


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(transport: str) -> int:
    if transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
    return 0


sys.exit(main())  # type: ignore[call-arg]
