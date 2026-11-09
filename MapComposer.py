"""Entry point. `MapComposer.exe` opens the app; `MapComposer.exe --selftest report.txt` checks the build."""
import sys
import multiprocessing


def _main():
    if "--selftest" in sys.argv:
        from mapcomposer.selftest import run
        i = sys.argv.index("--selftest")
        sys.exit(run(sys.argv[i + 1] if len(sys.argv) > i + 1 else None))
    from mapcomposer.app import main
    main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _main()
