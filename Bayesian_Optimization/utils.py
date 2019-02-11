import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)

data_dir = "data/mnist"

transforms_0 = transforms.Compose([transforms.ToTensor()])
train_dat_0_all = datasets.MNIST(
    data_dir, download=True, train=True, transform=transforms_0
)
train_dat_0, val_dat_0 = random_split(train_dat_0_all, [54000, 6000])
test_dat_0 = datasets.MNIST(data_dir, train=False, transform=transforms_0)

train_loader_0 = DataLoader(train_dat_0, batch_size=64, shuffle=True)
val_loader_0 = DataLoader(val_dat_0, batch_size=1000, shuffle=False)
test_loader_0 = DataLoader(test_dat_0, batch_size=1000, shuffle=False)


transforms_1 = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dat_1_all = datasets.MNIST(data_dir, train=True, transform=transforms_1)
train_dat_1, val_dat_1 = random_split(train_dat_1_all, [54000, 6000])
test_dat_1 = datasets.MNIST(data_dir, train=False, transform=transforms_1)

train_loader_1 = DataLoader(train_dat_1, batch_size=64, shuffle=True)
val_loader_1 = DataLoader(val_dat_1, batch_size=1000, shuffle=False)
test_loader_1 = DataLoader(test_dat_1, batch_size=1000, shuffle=False)


tmp = iter(train_loader_1)
sample_inputs, sample_targets = next(tmp)


def train(
    model,
    dataloader,
    optimizer,
    l1_penalty_coef,
    l2_penalty_coef,
    suppress_output,
):
    total_loss = 0.0
    correct = 0

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        total_loss += len(inputs) * loss.item()
        if l1_penalty_coef != 0.0:
            loss += l1_penalty_coef * model.l1_weight_penalty()
        if l2_penalty_coef != 0.0:
            loss += l2_penalty_coef * model.l2_weight_penalty()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    train_loss = total_loss / len(dataloader.dataset)
    train_accuracy = correct / len(dataloader.dataset)

    if not suppress_output:
        print(
            "Train set:\tAverage loss: {:.4f}, Accuracy: {:.4f}".format(
                train_loss, train_accuracy
            )
        )

    return train_loss, train_accuracy


def evaluate(model, dataloader, eval_type, suppress_output=True):
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            total_loss += F.nll_loss(outputs, targets, reduction="sum")
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    if not suppress_output:
        print(
            "{} set:\tAverage loss: {:.4f}, Accuracy: {:.4f}\n".format(
                eval_type, loss, accuracy
            )
        )

    return loss, accuracy


def run_experiment(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    n_epochs,
    l1_penalty_coef,
    l2_penalty_coef,
    suppress_output=True,
):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(n_epochs):
        if not suppress_output:
            print("Epoch {}: training...".format(epoch))
        train_loss, train_accuracy = train(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            l1_penalty_coef=l1_penalty_coef,
            l2_penalty_coef=l2_penalty_coef,
            suppress_output=suppress_output,
        )
        val_loss, val_accuracy = evaluate(
            model=model,
            dataloader=val_loader,
            eval_type="Validation",
            suppress_output=suppress_output,
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    final_test_loss, final_test_accuracy = evaluate(
        model=model,
        dataloader=test_loader,
        eval_type="Test",
        suppress_output=suppress_output,
    )

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy,
    }
