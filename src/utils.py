import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_confusion_matrix(cm: np.ndarray, model_name: str, filepath: str):
    """
    Saves a confusion matrix plot to disk so it can be logged as an MLflow artifact.
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
