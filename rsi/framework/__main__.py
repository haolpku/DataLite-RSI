"""Allow ``python -m rsi.framework --task ...``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
