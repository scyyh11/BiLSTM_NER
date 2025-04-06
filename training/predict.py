
import torch

def predict(model, dataloader, idx_to_label, device):
    model.eval().to(device)
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            for sent in preds:
                predictions.append([idx_to_label[idx] for idx in sent])
    return predictions
