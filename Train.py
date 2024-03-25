import os

import torch
from time import time
from sklearn import metrics

from model import device
from Datasets import file_Path


def train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, val_freq, PATH):
    since = time()
    best_acc = 0
    model_dir = os.path.join(file_Path, PATH)

    for e in range(epochs):
        print('Epoch{}/{}'.format(e + 1, epochs))
        print('-' * 10)
        count = 0
        train_loss = 0
        accuracy = 0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            predicted = torch.argmax(outputs, dim=1).cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            count += 1
            train_loss += loss.item() * images.size(0)
            accuracy += metrics.accuracy_score(labels, predicted)

            print('{} Loss: {:.4f}, {} Accuracy: {:.4f}'.format('train', train_loss / count, 'train', accuracy / count))

        if (e + 1) % val_freq == 0:
            print("Validation : ")
            with torch.no_grad():
                model.eval()
                val_loss = 0
                val_count = 0
                val_accuracy = 0

                for i, (images, labels) in enumerate(val_dataloader):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predicted = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()

                    val_count += 1
                    val_accuracy += metrics.accuracy_score(labels, predicted)

            print('{} Loss: {:.4f}, {} Accuracy: {:.4f}'.format('validation', val_loss / val_count,
                                                                'validation', val_accuracy / val_count))

            if (val_accuracy / val_count) > best_acc:
                best_acc = (val_accuracy / val_count)
                torch.save(model.state_dict(), f"{model_dir}/Epoch_{e + 1}.pth")

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model
