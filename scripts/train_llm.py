from llm.config import read_config
from llm.trainer import Trainer


def main():
    config = read_config("configs/config.yaml")
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    # not use flash attention: 1.5
    # use flash attention 1.73
    main()