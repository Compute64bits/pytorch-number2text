import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from os import path

# hyperparameters
batch_size = 12  # number of image per batch
leaning_rate = 0.1  # learning rate
print_every = 1000  # number of iteration before print information
# -----------------

# transforms.Compose = a list of transforms; 
# transforms.ToTensor = convert PIL image to tensor; 
# transforms.Normalize = normalize the image
# transforms.Normalize((0.1307,), (0.3081,)) = "mean" and "std" of the MNIST dataset (it's give by the documentation)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# download / load the MNIST dataset
# number of image : 60000 (train) + 10000 (test) = 70000
datasets_train = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
datasets_test = datasets.MNIST(root='../data', train=False, download=True, transform=transform)


# load the data
train_loader = torch.utils.data.DataLoader(datasets_train, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(datasets_test, batch_size=batch_size)

# 784 = 28px * 28px (image size)
# 10 = number of classes/output (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# matrix : width = 784; height = 10
def create_weights():
    return torch.randn(784, 10, requires_grad=True)


def test(weights, text_loader):
    test_size = len(text_loader.dataset)
    correct = 0

    # batch_idx = index of the batch (id)
    # data = image  ;  target = label (return number can show on the image)
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.view((-1, 28*28))  # flatten the image, img 28x28 to 784 in one line

        # output[0] = first img, output[1] = second img, ...
        # output.shape = (32, 10) (32 = batch_size, 10 = number of classes)
        output = torch.matmul(data, weights) 

        # ---> number to probability

        # dim = 1, because we want to apply softmax on the second dimension (the 10 classes)
        # dim = 1, because : output.shape = (32, 10) (32 = batch_size, 10 = number of classes)
        softmax = F.softmax(output, dim=1)

        # highest probability (example : 0.9 >>> 0.1)
        # keepdim = True, because we want to keep the dimension (32, 10) (32 = batch_size, 10 = number of classes)
        pred = softmax.argmax(dim=1, keepdim=True)
        # pred[0].item() = number of the prediction

        # number of correct prediction on this batch
        # target.view_as(pred)          = convert the target to the same shape as pred
        # pred.eq(target.view_as(pred)) = compare the prediction with the target
        # .sum()                        = sum (+) the number of correct prediction
        # .item()                       = convert the tensor to a number
        n_correct = pred.eq(target.view_as(pred)).sum().item()

        correct += n_correct
    
    acc = (correct / test_size) * 100  # moyenne of correct prediction
    print("\nAccuracy :", str(acc) + "%")


def train(weights, train_loader):
    it = 0
    for batch_idx, (data, targets) in enumerate(train_loader):

        # reset buffer gradient if exist
        if weights.grad is not None:
            weights.grad.zero_() 

        data = data.view((-1, 28*28))  # flatten the image, img 28x28 to 784 in one line

        # output[0] = first img, output[1] = second img, ...
        # output.shape = (32, 10) (32 = batch_size, 10 = number of classes)
        output = torch.matmul(data, weights) 

        # ---> calcul the loss (error)
        log_softmax = F.log_softmax(output, dim=1)  # dim = 1, because we want to apply softmax on the second dimension (the 10 classes)
        loss = F.nll_loss(log_softmax, targets)

        loss.backward()

        with torch.no_grad():
            weights -= leaning_rate * weights.grad

        it += 1
        if it % print_every == 0:
            print("Iteration :", it, "; Loss :", loss.item())

    print("Iteration :", it, "; Loss :", loss.item())


def save(weights, output_path: str = None):
    if output_path is None:
        output_path = str(path.basename(path.realpath(__file__))) + ".pth"
    print("\nOutput path :", output_path)
    torch.save(weights, output_path)
    print("Weights saved !")

def load(weights_path: str):
    print("Weights path :", weights_path)
    weights = torch.load(weights_path)
    print("Weights loaded !")
    return weights


weights = create_weights()
train(weights, train_loader)
test(weights, test_loader)
save(weights)
# load(weights, "number_finder.py.pth")
