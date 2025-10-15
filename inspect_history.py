import argparse
import pandas as pd
import matplotlib.pyplot as plt

import torch


def main(args):
    print("Loading checkpoint")
    checkpoint = torch.load(args.training_artifact, weights_only=False)
    print("loading compelte")
    batch_history = checkpoint["batch_history"]
    epoch_history = checkpoint["epoch_history"]

    loss = epoch_history["loss"]
    batch_loss_avg = batch_history["loss_avg"]

    df = pd.DataFrame(epoch_history)
    train_history = df[df["mode"] == "train"]
    val_history = df[df["mode"] == "validate"]
    # plt.plot(batch_history)
    # plt.show()

    plt.plot(train_history["loss"], label="Train")
    plt.plot(val_history["loss"], label="Val")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ta", "--training_artifact", required=True)
    args = parser.parse_args()

    main(args)
