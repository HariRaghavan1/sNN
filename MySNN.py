
import os
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader
import snntorch as snn

class PreprocessedDVSGesture(TorchDataset):
   def __init__(self, data_dir):
       self.data_dir = data_dir
       self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
       self.files.sort()

   def __len__(self):
       return len(self.files)

   def __getitem__(self, idx):
       file_path = os.path.join(self.data_dir, self.files[idx])
       npzfile = np.load(file_path)
       data = npzfile['data']
       label = npzfile['label']
       return torch.tensor(data, dtype=torch.float32), label

# Define the SNN model
class SNN(nn.Module):
   def __init__(self, num_bins):
       super(SNN, self).__init__()
       self.num_bins = num_bins
       self.conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
       self.pool1 = nn.MaxPool2d(2, 2)
       self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)  # Reduced from 32 to 4 filters
       self.pool2 = nn.MaxPool2d(2, 2)
       self.fc1 = nn.Linear(4 * 32 * 32, 128)  # Reduced neuron count in the first fully connected layer
       self.fc2 = nn.Linear(128, 11)  # Output layer remains connected to the number of output classes
       self.lif1 = snn.Leaky(beta=0.95)
       self.lif2 = snn.Leaky(beta=0.95)
       self.lif3 = snn.Leaky(beta=0.95)

   def forward(self, x):
       batch_size, num_bins, height, width, channels = x.size()
       x = x.permute(0, 1, 4, 2, 3).contiguous()  # Change to [batch, bins, channels, height, width]
       x = x.view(batch_size * num_bins, channels, height, width)

       cur1 = self.conv1(x)
       spk1, mem1 = self.lif1(cur1)
       cur2 = self.pool1(mem1)
       cur3 = self.conv2(cur2)
       spk3, mem3 = self.lif2(cur3)
       cur4 = self.pool2(mem3)
       cur4 = cur4.view(batch_size * num_bins, -1)
       cur4 = nn.Flatten()(cur4)  # Add a Flatten layer before fc1
       cur5 = self.fc1(cur4)
       spk5, mem5 = self.lif3(cur5)
       out = self.fc2(spk5)
       out = out.view(batch_size, num_bins, -1)
       return out.sum(dim=1)  # Summing over the temporal dimension

# Paths to preprocessed data
preprocessed_train_save_dir = '/Users/hariraghavan/Downloads/DvsGesture/preprocessed_training'
preprocessed_test_save_dir = '/Users/hariraghavan/Downloads/DvsGesture/preprocessed_testing'

# Load datasets
train_dataset = PreprocessedDVSGesture(preprocessed_train_save_dir)
test_dataset = PreprocessedDVSGesture(preprocessed_test_save_dir)

# Create data loaders
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine the number of bins (assuming the first file's shape is representative)
num_bins = np.load(os.path.join(preprocessed_train_save_dir, os.listdir(preprocessed_train_save_dir)[0]))['data'].shape[0]

# Instantiate the model, loss function, and optimizer
model = SNN(num_bins=num_bins).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(num_epochs):
   best_accuracy = 0.0
   for epoch in range(num_epochs):
       running_loss = 0.0
       print(f"Epoch {epoch + 1}/{num_epochs} started.")
       model.train()
       for i, (inputs, labels) in enumerate(train_loader):
           inputs = inputs.to(device)
           labels = labels.to(device)

           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, labels)

           running_loss += loss.item()
           if i % 10 == 9:
               print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
               running_loss = 0.0

       # Backpropagate and update weights after all batches have been processed
       loss.backward()
       optimizer.step()

       # Evaluate on the test set
       print("Evaluating on the test set...")
       accuracy = test_accuracy()
       if accuracy > best_accuracy:
           best_accuracy = accuracy
           torch.save(model.state_dict(), 'best_model.pth')
       print(f"Epoch {epoch + 1} completed. Best accuracy: {best_accuracy:.2f}%")

# Testing loop
def test_accuracy():
   model.eval()
   correct = 0
   total = 0
   with torch.no_grad():
       for inputs, labels in test_loader:
           inputs = inputs.to(device)
           labels = labels.to(device)
           outputs = model(inputs)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()

   accuracy = 100 * correct / total
   print(f"Accuracy on the test set: {accuracy:.2f}%")
   return accuracy

# Train the model
print("Starting training process...")
train(num_epochs=10)

# Load the best model and evaluate on the test set
print("Loading the best model and evaluating on the test set...")
model.load_state_dict(torch.load('best_model.pth'))
test_accuracy()
print("Training and testing process completed.")
