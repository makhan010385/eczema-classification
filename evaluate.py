import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from tqdm import tqdm
import os
import config
from model import get_model
from data_preprocessing import get_transforms

# Training used only the first 50 images per class in os.listdir order
# (create_data_loaders default). Evaluating on files beyond that index in the
# same listing order guarantees unseen data.
TRAIN_LIMIT = 50
EVAL_PER_CLASS = 200


def get_heldout_paths():
    eczema_files = [f for f in os.listdir(config.ECZEMA_PATH)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][TRAIN_LIMIT:TRAIN_LIMIT + EVAL_PER_CLASS]
    normal_files = [f for f in os.listdir(config.NORMAL_PATH)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][TRAIN_LIMIT:TRAIN_LIMIT + EVAL_PER_CLASS]

    paths = ([os.path.join(config.ECZEMA_PATH, f) for f in eczema_files] +
             [os.path.join(config.NORMAL_PATH, f) for f in normal_files])
    labels = [1] * len(eczema_files) + [0] * len(normal_files)
    return paths, labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = get_model('resnet').to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    paths, labels = get_heldout_paths()
    print(f"Evaluating on {len(paths)} held-out images "
          f"({sum(labels)} eczema, {len(labels) - sum(labels)} normal)")

    transform = get_transforms('test')
    preds, probs, confidences = [], [], []

    with torch.no_grad():
        for path in tqdm(paths, desc="Evaluating"):
            image = cv2.imread(path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = transform(image=image)['image'].unsqueeze(0).to(device)
            p = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
            preds.append(int(p.argmax()))
            probs.append(float(p[1]))
            confidences.append(float(p.max()))

    targets = labels[:len(preds)]
    report = classification_report(targets, preds,
                                   target_names=config.CLASS_NAMES,
                                   output_dict=True)
    metrics = {
        'accuracy': float(accuracy_score(targets, preds)),
        'precision': float(precision_score(targets, preds)),
        'recall': float(recall_score(targets, preds)),
        'f1': float(f1_score(targets, preds)),
        'auc': float(roc_auc_score(targets, probs)),
        'n_eval': len(preds),
        'per_class': {
            name: {
                'precision': float(report[name]['precision']),
                'recall': float(report[name]['recall']),
                'f1': float(report[name]['f1-score']),
            } for name in config.CLASS_NAMES
        },
        'mean_confidence': float(np.mean(confidences)),
        'confidences': confidences,
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    fpr, tpr, _ = roc_curve(targets, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(json.dumps(metrics, indent=2))
    print("Saved metrics.json, confusion_matrix.png, roc_curve.png")


if __name__ == "__main__":
    main()
