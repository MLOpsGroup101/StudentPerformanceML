Generating the docs
----------

This project uses [MkDocs](https://www.mkdocs.org/) with the configuration file
[docs/mkdocs.yml](mkdocs.yml) and Markdown sources in [docs/source/](source/).

Build the documentation locally from the project root with:

    uv run mkdocs build --config-file docs/mkdocs.yml --site-dir build

or via the invoke task:

    uv run invoke build-docs

Serve the documentation locally with live reload:

    uv run mkdocs serve --config-file docs/mkdocs.yml

or via the invoke task:

    uv run invoke serve-docs

