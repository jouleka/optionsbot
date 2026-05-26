"""Entry point for the `optionsbot-daemon` console script."""
import sys


def main() -> int:
    sys.stderr.write(
        "optionsbot-daemon is not yet implemented. See IBK-7 for the implementation plan.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
