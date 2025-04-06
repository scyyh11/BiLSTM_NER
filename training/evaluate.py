import torch
from sklearn.metrics import precision_recall_fscore_support


def evaluate_model(model, dataloader, device):
    model.eval().to(device)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(targets.view(-1).cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=1.0)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
