import os
import sys


def main():
    from src.gui_tuner import run_app

    folder = os.path.join('data', 'SBU-shadow', 'SBUTrain', 'ShadowImages')
    if len(sys.argv) > 1:
        folder = sys.argv[1]

    run_app(folder)


if __name__ == '__main__':
    main()
