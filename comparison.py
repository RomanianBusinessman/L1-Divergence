
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Teacher Model
class TeacherCNN(nn.Module):
    def __init__(self):
        super(TeacherCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 14 * 14, 256)  
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 14 * 14)  
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Student Model 
class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.fc1 = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 16 * 26 * 26)
        x = self.fc1(x)
        return x

# Data Loading
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Train Teacher
def train_teacher(model, epochs=5):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Shape checks
            assert images.shape[0] == labels.shape[0], f"Batch size mismatch: images {images.shape[0]}, labels {labels.shape[0]}"

            optimizer.zero_grad()
            outputs = model(images)
            assert outputs.shape[0] == labels.shape[0], f"Output batch size {outputs.shape[0]} != labels {labels.shape[0]}"

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Teacher Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

    # Evaluate teacher
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Teacher Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Distillation Training
def distill(student, teacher, loss_type='kl', epochs=5, T=5.0, alpha=0.9):
    student.to(device)
    teacher.to(device)
    student.train()
    teacher.eval()
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    criterion_ce = nn.CrossEntropyLoss()

    if loss_type == 'kl':
        criterion_distill = nn.KLDivLoss(reduction='batchmean')
    else:
        criterion_distill = None  # L1 or L2 handled inline

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Shape checks
            assert images.shape[0] == labels.shape[0], f"Batch size mismatch in distill: images {images.shape[0]}, labels {labels.shape[0]}"

            optimizer.zero_grad()
            teacher_logits = teacher(images)
            student_logits = student(images)

            # Soft targets
            teacher_probs = torch.softmax(teacher_logits / T, dim=1)
            student_probs = torch.softmax(student_logits / T, dim=1)
            student_log_probs = torch.log_softmax(student_logits / T, dim=1)

            # Distillation loss
            if loss_type == 'kl':
                loss_distill = criterion_distill(student_log_probs, teacher_probs) * (T * T)
            elif loss_type == 'l1':
                loss_distill = torch.sum(torch.abs(teacher_probs - student_probs), dim=1).mean() * (T * T)
            elif loss_type == 'l2':
                loss_distill = torch.sum((teacher_probs - student_probs) ** 2, dim=1).mean() * (T * T)

            # Cross-entropy loss
            loss_ce = criterion_ce(student_logits, labels)

            # Combined loss
            loss = alpha * loss_distill + (1 - alpha) * loss_ce
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"{loss_type.upper()} Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

    # Evaluate student
    student.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion_ce(outputs, labels).item()
    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f"{loss_type.upper()} Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}")
    return accuracy, avg_test_loss

# Main Experiment
if __name__ == '__main__':
    # Train teacher
    teacher = TeacherCNN()
    teacher_accuracy = train_teacher(teacher, epochs=5)

    # Save teacher model
    torch.save(teacher.state_dict(), 'teacher_mnist.pth')

    # Distillation with different losses
    losses = ['kl', 'l1', 'l2']
    results = {}

    for loss_type in losses:
        print(f"\nDistilling with {loss_type.upper()} Loss")
        student = StudentCNN()
        accuracy, test_loss = distill(student, teacher, loss_type=loss_type, epochs=5, T=5.0, alpha=0.9)
        results[loss_type] = {'accuracy': accuracy, 'test_loss': test_loss}
        torch.save(student.state_dict(), f'student_mnist_{loss_type}.pth')

    # Print results
    print("\nSummary of Results:")
    for loss_type in losses:
        print(f"{loss_type.upper()}: Accuracy = {results[loss_type]['accuracy']:.2f}%, Test Loss = {results[loss_type]['test_loss']:.4f}")
