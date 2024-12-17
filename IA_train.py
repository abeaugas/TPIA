import torch as pt
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
from utils.AudioDataset import AudioDataset
from utils.TD_network import CNNNetwork

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
    accuracies = []
    percentages = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    for percent in percentages:
        # Charger les données
        if OUTSIZE:
            labels = {"rain": 0, "walking":1}
            trainData = AudioDataset("meta/bdd_A_train.csv", labels)
            testData = AudioDataset("meta/bdd_A_test.csv", labels)
        else:
            labels = {"rain": 0, "walking":1, "wind": 2, "car_passing": 3}
            trainData = AudioDataset("meta/bdd_B_train.csv", labels,frac_=percent)
            testData = AudioDataset("meta/bdd_B_test.csv", labels)
            model.load_state_dict(pt.load('best_model.pth'))
        
        train_loader = pt.utils.data.DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = pt.utils.data.DataLoader(testData)


        print(f"\n### Training with: {percent * 100}% ###")
        print(f"Total number of parameters: {model.count_parameters()}")

        #######################
        # Entraîner le modèle #
        #######################
        accuracies.append(train(model, train_loader, test_loader, loss_fn, optimizer, device, EPOCHS, PATIENCE))
    for val in percentages:
        val *= 100
    plt.plot(percentages, accuracies)
    plt.ylim(0, 100)
    plt.show()



    ############################################################################
    # Charger le meilleur modèle sauvegardé et évaluer sur les données de test #
    ############################################################################
    # testData = AudioDataset("meta/bdd_B_test.csv", labels)
    # test_loader = pt.utils.data.DataLoader(testData)
    # # Charger le meilleur modèle sauvegardé
    model.load_state_dict(pt.load('best_model.pth', weights_only=True))


    # # Évaluer le modèle sur les données de test
    # test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device, "TestB")
    # print(f"TestB Loss: {test_loss:.4f}, TestB Accuracy: {test_accuracy:.2f}%")



