import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import transforms
import torchvision
from model import VICReg
from tqdm import tqdm

# load model checkpoint and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cp = torch.load("checkpoint.pt", map_location=device) 
encoder_dim, projector_dim = cp["encoder_dim"], cp["projector_dim"]
model = VICReg(encoder_dim, projector_dim).to(device)
model.load_state_dict(cp["model_state_dict"])
model.eval()

# create linear layer, optimizer, scheduler and training hyperparams
num_classes, batch_size, num_epochs = 10, 256, 10

#TODO: LINEAR CLASSIFIER
#linear = 

opt = SGD(linear.parameters(), lr=0.02, weight_decay=1e-6)
scheduler = CosineAnnealingLR(opt, num_epochs)

# data augmentations used to regularize the linear layer
augment = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
augment_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((32,32))
])

# define train and test datasets
cifar_train_data = CIFAR10(root=".", train=True, download=True, transform=augment)
cifar_test_data = CIFAR10(root=".", train=False, download=True, transform=augment_test)
cifar_train_dataloader = DataLoader(cifar_train_data, batch_size, shuffle=True)
cifar_test_dataloader = DataLoader(cifar_test_data, batch_size)

# use standard cross entropy loss
criterion = nn.CrossEntropyLoss()
progress = tqdm(range(num_epochs))


for _ in progress:
    for images, labels in cifar_train_dataloader:
        images, labels = images.to(device), labels.to(device)
        encoder_out = model.encoder(images)

        #TODO: LINEAR CLASSIFIER 
        #preds = 

        loss = criterion(preds, labels)
        loss.backward()
        opt.step()
        opt.zero_grad()
        progress.set_description(f"Loss: {loss.item()}")
    scheduler.step()

# evaluate the accuracy 
num_correct = len(cifar_test_data)
for images, labels in cifar_test_dataloader:
    images, labels = images.to(device), labels.to(device)
    encoder_out = model.encoder(images)
    
    #TODO: LINEAR CLASSIFIER
    #preds =
    
    num_incorrect = torch.count_nonzero(preds-labels)
    num_correct -= num_incorrect
print("Accuracy:", num_correct / len(cifar_test_data))
