import torch as pt
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import statistics
import numpy as np
from math import sqrt
from scipy.stats import entropy
from utils.AudioDataset import AudioDataset
from utils.TD_network import CNNNetwork


def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.02):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval


def least_confidence(net,test_data,n):
  """
  Returns the sample index from test_data based on the uncertainty scores
  using the least confidence sampling

  Keyword arguments:
    net       : the trained neural network
    test_data : the samples as the input of the net
    n         : number of the best candidates based on least confidence

  Output:
    res.keys(): a list of indexes from input samples (test_data)
  """
  uncertainty_dict={} # as sample_index: uncertainty score
  for i,(x,y) in enumerate(test_data):
    preds = net(pt.unsqueeze(x,0)) # preds is a probability distribution of classes
    uncertainty_dict[i]=1-np.max(preds[0].cpu().detach().numpy())
  res = dict(sorted(uncertainty_dict.items(),
                    key = lambda x: x[1], reverse = True)[:n])
  return res.keys()

def margin_confidence(net,test_data,n):
  """
  Returns the sample index from test_data based on the uncertainty scores
  using the margin of confidence sampling

  Keyword arguments:
    net       : the trained neural network
    test_data : the samples as the input of the net
    n         : number of the best candidates based on margin of confidence

  Output:
    res.keys(): a list of indexes from input samples (test_data)
  """
  uncertainty_dict={} # as sample_index: uncertainty score
  for i,(x,y) in enumerate(test_data):
    preds = net(pt.unsqueeze(x,0)) # preds is a probability distribution of classes
    preds_sorted = np.sort(preds[0].cpu().detach().numpy())
    uncertainty_dict[i]=preds_sorted[0]-preds_sorted[1]
  res = dict(sorted(uncertainty_dict.items(),
                    key = lambda x: x[1], reverse = True)[:n])
  return res.keys()

def ratio_confidence(net,test_data,n):
  """
  Returns the sample index from test_data based on the uncertainty scores
  using the ratio of confidence sampling

  Keyword arguments:
    net       : the trained neural network
    test_data : the samples as the input of the net
    n         : number of the best candidates based on ratio of confidence

  Output:
    res.keys(): a list of indexes from input samples (test_data)
  """
  uncertainty_dict={} # as sample_index: uncertainty score
  for i,(x,y) in enumerate(test_data):
    preds = net(pt.unsqueeze(x,0)) # preds is a probability distribution of classes
    preds_sorted = np.sort(preds[0].cpu().detach().numpy())
    uncertainty_dict[i]=preds_sorted[0]/preds_sorted[1]
  res = dict(sorted(uncertainty_dict.items(),
                    key = lambda x: x[1], reverse = True)[:n])
  return res.keys()

def entropy_confidence(net,test_data,n):
  """
  Returns the sample index from test_data based on the uncertainty scores
  using the entropy of confidence sampling

  Keyword arguments:
    net       : the trained neural network
    test_data : the samples as the input of the net
    n         : number of the best candidates based on entropy of confidence

  Output:
    res.keys(): a list of indexes from input samples (test_data)
  """
  uncertainty_dict={} # as sample_index: uncertainty score
  for i,(x,y) in enumerate(test_data):
    preds = net(pt.unsqueeze(x,0)) # preds is a probability distribution of classes
    uncertainty_dict[i]=entropy(preds[0].cpu().detach().numpy())
  res = dict(sorted(uncertainty_dict.items(),
                    key = lambda x: x[1], reverse = True)[:n])
  return res.keys()


def train_single_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    with tqdm.tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backpropagate the loss and update the gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
    print("")


def evaluate(model, dataloader, loss_fn, device,name=""):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    model.eval()
   
    with tqdm.tqdm(total=len(dataloader), desc="Evaluation "+name, unit="batch") as pbar:
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = pt.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()
            pbar.update(1)

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy


def train(model, train_loader, test_loader, loss_fn, optimizer, device, epochs, patience):
    best_loss = float('inf')
    best_epoch = -1
    best_accuracy = 0

    patience_counter = 0

    print("")
    print('-------------------------------------------')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_single_epoch(model, train_loader, loss_fn, optimizer, device)
        

        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device, "Test")

        if best_accuracy < test_accuracy:
            best_loss = test_loss
            patience_counter = 0
            best_accuracy = test_accuracy
            best_epoch = epoch
            # Sauvegarder le meilleur modèle
            pt.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}% (Best at {best_epoch+1}: {best_accuracy:.2f}%)")

        if patience_counter >= patience:
            print("Early stopping triggered")
            break
        print('-------------------------------------------')
        print("")
    print('Finished Training')
    return(best_accuracy)


if __name__ == "__main__":
    ##########################
    ##### Hyperparamètres ####
    ##########################
    BATCH_SIZE = 1           # Taille du lot
    EPOCHS = 10              # Nombre d'époques
    PATIENCE = 10           # Nombre d'époques sans amélioration avant l'arrêt
    OUTSIZE = False           # True pour DA, False pour DB
    ##########################
    ##########################


    #################
    # Initilisation #
    #################
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # Initialiser le modèle, la fonction de perte et l'optimiseur
    model = CNNNetwork().to(device)
    model.outsize = OUTSIZE
    print("Charger pour 2 labels" if OUTSIZE else "Charger pour 4 labels")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Boucler pour tester plusieurs pourcentages
    percentages = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    accuracies = []
    for percent in percentages:
        # Charger les données
        if OUTSIZE:
            labels = {"rain": 0, "walking":1}
            trainData = AudioDataset("meta/bdd_A_train.csv", labels)
            testData = AudioDataset("meta/bdd_A_test.csv", labels)
        else:
            labels = {"rain": 0, "walking":1, "wind": 2, "car_passing": 3}
            trainData = AudioDataset("meta/bdd_B_train.csv", labels)
            uncertaintyDataIndices = list(entropy_confidence(model, trainData, int(len(trainData.data_frame)*percent))) # Sélectioner les données avec uncertainty sampling
            trainData = AudioDataset("meta/bdd_B_train.csv", labels, confident_list=uncertaintyDataIndices)
            testData = AudioDataset("meta/bdd_B_dev.csv", labels)
            model.load_state_dict(pt.load('BestModelSave/best_model.pth'))
        
        train_loader = pt.utils.data.DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = pt.utils.data.DataLoader(testData)


        print(f"\n### Training with: {percent * 100}%  ###")
        print(f"Total number of parameters: {model.count_parameters()}")

        #######################
        # Entraîner le modèle #
        #######################
        train(model, train_loader, test_loader, loss_fn, optimizer, device, EPOCHS, PATIENCE)

        ############################################################################
        # Charger le meilleur modèle sauvegardé et évaluer sur les données de test #
        ############################################################################
        testData = AudioDataset("meta/bdd_B_test.csv", labels)
        test_loader = pt.utils.data.DataLoader(testData)
        # Charger le meilleur modèle sauvegardé
        model.load_state_dict(pt.load('best_model.pth', weights_only=True))


        # Évaluer le modèle sur les données de test
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device, "TestB")
        accuracies.append(test_accuracy)
        print(f"TestB Loss: {test_loss:.4f}, TestB Accuracy: {test_accuracy:.2f}%")


    print(accuracies)
    print(percentages)
    plt.plot(percentages, accuracies, marker='o')
    plt.title('Accuracies')
    plt.ylim(0, 100)
    plt.show()