import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import sys
import os
import json
from model import resnet34

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_dataset = datasets.ImageFolder(root="./test",
                                         transform=data_transform)
    test_num = len(test_dataset)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size, shuffle=False,
                                               num_workers=nw)

    print("using {} images for testing.".format(test_num))

    net = resnet34()
    # load saved weights
    model_weight_path = "./resNet34.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 515)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()
    epochs = 25
    for epoch in range(epochs):
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

                test_bar.desc = "test"

        test_accurate = acc / test_num
        print('test_accuracy: %.3f' % (test_accurate))
    # loss_train = history.history['train_loss']



if __name__ == '__main__':
    main()
