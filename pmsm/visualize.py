import argparse
from preprocessing import file_utils as futils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize performance of the '
                                                 'given model uid.')
    parser.add_argument('-t', '--trunc', action='store_true',
                        help='Truncate the time-series at a time step t ('
                             'fixed, where both testsets are separated)')
    parser.add_argument('model_uid',
                        help='The 6-digit model uid in hex')

    args = parser.parse_args()

    truncate_at = 40092 if args.trunc else None

    report = futils.Report.load(args.model_uid, truncate_at=truncate_at)
    report.print()
    try:
        report.plot()
    except Exception:
        print('plot failed')
