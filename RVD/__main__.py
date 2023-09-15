import argparse


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
        help=''
    )


    
    return parser.parse_args()



def main():
    """
    This is a script for running the pre-trained voicing detection model by taking WAV file(s) as input. 
    """
    args = parse_args()




# entry point
main()
