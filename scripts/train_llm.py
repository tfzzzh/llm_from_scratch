import argparse
from llm.config import read_config
from llm.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    args = parser.parse_args()

    
    config = read_config(args.config_file)
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    # not use flash attention: 1.5
    # use flash attention 1.73
    main()