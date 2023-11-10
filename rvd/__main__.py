import argparse
from .core import Model


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument(
        '--model_file',
        type=str,
        default='./rvd/pretrained/rvd_weights_revised.pth',
        help='Path of the model file'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default='0',
        help='The gpu to perform inference on')
    parser.add_argument(
        '--voicing_threshold',
        type=float,
        default='0.5',
        help='The threshold used for deciding voicing.'
    )
    # parser.add_argument(
    #     '--audio_file',
    #     nargs='+',
    #     required=True,
    #     help='The audio file to process')
    # parser.add_argument(
    #     '--output_files',
    #     nargs='+',
    #     required=True,
    #     help='The file to save voicing decisions.')
    return parser.parse_args()



def main(filename):
    """
    This is a script for running the pre-trained voicing detection model by taking WAV file(s) as input. 
    """
    args = parse_args()

    # Get inference device
    device = 'cpu' if args.gpu is None else f'cuda:{args.gpu}'

    rvd = Model(args)
    print("here")

    # filename = './rvd/example/rl012.wav'
    rvd.predict()

# entry point
main()