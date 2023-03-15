import argparse

import torch

from torchvision import datasets

from utils import Preprocessor
from model import Net
from train import Trainer
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='mnist classification')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    pp = Preprocessor()
    
    # model
    model = Net()
    
    #LFW Datasets
    path = "data"
    
    #Pairs Data
    training_pair = datasets.LFWPairs(
        root=path,
        split="train",
        transform=pp.train_tran,
        download=True
    )
    
    test_pair = datasets.LFWPairs(
        root=path,
        split="test",
        transform=pp.test_tran,
        download=True
    )
    
    #People Data
    train_Loader = datasets.LFWPeople(
        root=path,
        split="train",
        transform=pp.train_tran,
        download=True
    )
    
    test_Loader = datasets.LFWPeople(
        root=path,
        split="test",
        transform=pp.test_tran,
        download=True
    )
    
    #print(torch.Tensor.size(train_Loader[1][0]))
    
    #showPeople(train_Loader)
    
    # trainer
    trainer = Trainer(model=model)
    
    # model training
    trainer.train(train_loader=train_Loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")
    
    return
    
def showPairs(data, cols, rows, shift=0): 
    """
    Parameters
    ----------
    data : Tensors 
        The Tensor data of the dataset, this only works for pairs
    cols : TYPE
        Number of pair columns to display
    rows : TYPE
        Number of pair rows to display.
    shift : TYPE, optional
        The starting index of the display. The default is 0.

    Returns
    -------
    None. The plot of images is displayed in the plots tab. (For Spyder Run Only)

    """
    
    
    figure = plt.figure()
    cols = cols*2
    for i in range(1, cols*rows +1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img1 = data[shift + i][0]
        img2 = data[shift + i][1]
        #Show image 1
        figure.add_subplot(rows, cols, i*2-1)
        plt.axis("off")
        plt.title("#%s: A" % (shift + i))
        plt.imshow(img1.permute(1,2,0))
        #plt.imshow(img1)
        #Show image 2
        figure.add_subplot(rows, cols, i*2)
        plt.axis("off")
        plt.title("#%s: B" % (shift + i))
        plt.imshow(img2.permute(1,2,0))
        #plt.imshow(img2)
        
        plt.title("Value: %s" % data[shift + i][2])
    plt.show()

def showPeople(data, cols=3, rows=3):
    figure = plt.figure(figsize=(8, 8))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        #plt.title(data[label])
        plt.axis("off")
        plt.imshow(img.permute(1,2,0))
    plt.show()

if __name__ == "__main__":
    main()
