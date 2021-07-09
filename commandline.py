
import sys
import getopt
from configs import data_config, train_config


def parse_args(argv):
    try:
        opts, args = getopt.getopt(
            argv, "t:e:d:w:p:c:s:l:", ["trainbs=", "epochs=", "datadir=", "workers=", "checkpointdir=", "continue=", "epochsave=", "logdir="])
    except getopt.GetoptError:
        print(
            'main.py -t <train_batch_size> -e <epochs> -d <datadir> -w <workers> -p <checkpointdir> -c <continue> -s <epochsave> -l <logdir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'main.py -t <train_batch_size> -e <epochs> -d <datadir> -w <workers> -p <checkpointdir> -c <continue> -s <epochsave> -l <logdir>')
            sys.exit()
        elif opt in ("-t", "--trainbs"):
            data_config.train_batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            train_config.epochs = int(arg)
        elif opt in ("-d", "--datadir"):
            data_config.data_dir = arg
        elif opt in ("-w", "--workers"):
            data_config.num_workers = int(arg)
        elif opt in ("-p", "--checkpointdir"):
            train_config.checkpoint_path = arg
        elif opt in ("-c", "--continue"):
            train_config.continue_training = True
        elif opt in ("-s", "--epochsave"):
            train_config.checkpoint_epochs = int(arg)
        elif opt in ("-l", "--logdir"):
            train_config.log_dir = arg
