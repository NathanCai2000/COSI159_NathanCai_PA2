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
    
    """---
    # datasets
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.1307,), (0.3081,))
    # ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )

    # trainer
    trainer = Trainer(model=model)
    
    # model import and test
    ---"""
    
    """
    import os
    # loads the model again and then evaluate itself again (Should have same scores)
    trainer.load_model(os.path.join("./save/", "mnist.pth"))
    trainer.eval(test_loader=test_loader)
    """
    
    """---
    # model training
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")

    # model evaluation
    trainer.eval(test_loader=test_loader)

    # util that loads the test data for use as the samples for testing model inference
    examples = enumerate(test_loader)
    tmp, (example_data, example_targets) = next(examples)
    example_data.shape
    
    
    # model inference
    sample = example_data  # complete the sample here
    trainer.infer(sample=sample)
    
    

    return
    ---"""
    
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
