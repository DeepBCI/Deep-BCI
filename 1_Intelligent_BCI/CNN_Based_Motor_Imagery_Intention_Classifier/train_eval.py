import torch
import torch.nn.functional as F
import utils
import time

def train(log_interval, model, device, train_loader, optimizer, scheduler, cuda, gpuidx, epoch=1):
    lossfn = torch.nn.CrossEntropyLoss()
    correct = []
    start = time.time()
    model.train()
    t_data = []
    t_model = []
    t3 = time.time()

    for batch_idx, datas in enumerate(train_loader):
        data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)
        t2 = time.time()
        t_data.append(t2 - t3)
        optimizer.zero_grad()
        output = model(data)
        pred = F.log_softmax(output, dim=1).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct.append(pred.eq(target.view_as(pred)).sum().item())
        loss = lossfn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        t3 = time.time()
        t_model.append(t3 - t2)

    print("time :", time.time() - start)
    print(f"t_data : {sum(t_data)} , t_model : {sum(t_model)}")
    print(f'Train set: Accuracy: {sum(correct)}/{len(train_loader.dataset)} ({100. * sum(correct) / len(train_loader.dataset):.4f}%)')


def eval(model, device, test_loader):
    model.eval()
    test_loss = []
    correct = []
    preds = []

    with torch.no_grad():
        for datas in test_loader:
            data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)
            output = model(data[:, :, :, :])
            test_loss.append(F.cross_entropy(output, target, reduction='sum').item())
            preds.append( F.softmax(output, dim=1).cpu().numpy())
            pred = F.log_softmax(output, dim=1).argmax(dim=1, keepdim=True)
            correct.append(pred.eq(target.view_as(pred)).sum().item())
            print(f'GT : {target}')
            print(f'Pred : {pred.squeeze()}')
            print(f'Correct : {pred.eq(target.view_as(pred)).squeeze()}')

    loss = sum(test_loss) / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))

    return loss, 100. * sum(correct) / len(test_loader.dataset)




