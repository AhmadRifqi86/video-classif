import torch
import time
import all_config
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support


def train_model(model, train_loader, criterion, optimizer, num_epochs=10, 
                save_model=True, early_stop=0.0):
    #print(f"Training with {allCONF_CLASSIF_MODE} classification mode")
    model.train()
    
    start = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(all_config.CONF_DEVICE), labels.to(all_config.CONF_DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if all_config.CONF_CLASSIF_MODE == "multiclass":
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else:
                batch_losses = []
                for i in range(outputs.size(1)):
                    output_i = outputs[:, i]
                    label_i = labels[:, i].float()
                    batch_losses.append(criterion[i](output_i, label_i))
                loss = sum(batch_losses)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.numel()
                correct += (predictions == labels).sum().item()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
              f"Accuracy: {epoch_acc:.4f}")

    duration = time.time() - start
    print(f"training_duration: {duration:.4f}")
    if save_model:
        torch.save(model, all_config.CONF_MODEL_PATH)
        print(f"Model saved to {all_config.CONF_MODEL_PATH}")

def evaluate_model(model, test_loader, class_names):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    start = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(all_config.CONF_DEVICE), labels.to(all_config.CONF_DEVICE)
            inputs = inputs.squeeze(2)
            outputs = model(inputs)

            if all_config.CONF_CLASSIF_MODE == "multiple_binary":
                predictions = torch.sigmoid(outputs) > 0.5
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            else:
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    if all_config.CONF_CLASSIF_MODE == "multiple_binary":
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # Calculate accuracy, precision, recall, and f1-score for each class
        accuracies = []
        for i, class_name in enumerate(class_names):
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels[:, i], all_predictions[:, i], average="binary")
            accuracy = np.mean(all_predictions[:, i] == all_labels[:, i])  # Calculate accuracy for this class
            accuracies.append(accuracy)
            print(f"Class {class_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, f1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        # Calculate overall precision, recall, and F1-score across all classes
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average="macro")
        print(f"Overall Precision: {overall_precision:.4f}, Overall Recall: {overall_recall:.4f}, Overall F1-Score: {overall_f1:.4f}")

        # Calculate and print overall accuracy across all classes
        overall_accuracy = np.mean(np.all(all_predictions == all_labels, axis=1))
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

    elif all_config.CONF_CLASSIF_MODE == "multiclass":
        accuracy = correct / total  # This is the overall accuracy
        print(f"Overall Accuracy: {accuracy:.4f}")

        # Class-wise precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)

        # Print per-class results
        for i, class_name in enumerate(class_names):
            print(f"Class: {class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, f1-Score: {f1[i]:.4f}")

        # Calculate overall precision, recall, and F1-score across all classes
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average="macro")
        print(f"Overall Precision: {overall_precision:.4f}, Overall Recall: {overall_recall:.4f}, Overall F1-Score: {overall_f1:.4f}")
    else:
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
    duration = time.time()-start
    print(f"inference_duration: {duration:.4f}")