import argparse
import rvd


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument(
        '--audio_files',
        nargs='+',
        required=True,
        help='The audio file to process')
    parser.add_argument(
        '--output_files',
        nargs='+',
        required=True,
        help='The file to save voicing decisions.')
    parser.add_argument(
        '--batch_size',
        type=int,
        help='The number of frames per batch')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The gpu to perform inference on')
    parser.add_argument(
        '--hop_length',
        type=int,
        help='The hop length of the analysis window')
    return parser.parse_args()



def main():
    """
    This is a script for running the pre-trained voicing detection model by taking WAV file(s) as input. 
    """
    args = parse_args()

    # Get inference device
    device = 'cpu' if args.gpu is None else f'cuda:{args.gpu}'

    rvd.predict(args.audio_files,
                args.output_files,
                args.hop_length,
                args.batch_size,
                device)

# entry point
main()
