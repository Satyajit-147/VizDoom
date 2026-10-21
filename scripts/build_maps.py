"""python scripts/build_maps.py maps/ -o build/ --variant buggy"""

import sys

from qatester.mapgen.cli import main

if __name__ == "__main__":
    main(["build", *sys.argv[1:]])
