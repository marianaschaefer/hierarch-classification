import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HierarchicalMTL(torch.nn.Module):
    def __init__(self, input_dim, n1_dim, n2_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU()
        )
        self.head_n1 = torch.nn.Linear(128, n1_dim)
        self.head_n2 = torch.nn.Linear(128, n2_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.head_n1(z), self.head_n2(z)

def avaliar():
    df = pd.read_csv("data_processed/dataset_MULTIMODAL_LIMPO.csv")
    le1, le2 = LabelEncoder(), LabelEncoder()
    le1.fit(df["classe_nivel1"])
    le2.fit(df["classe_nivel2"])
    
    cols_features = [c for c in df.columns if c not in ["supercells", "classe_nivel1", "classe_nivel2"]]
    X = StandardScaler().fit_transform(df[cols_features])
    y2 = le2.transform(df["classe_nivel2"])

    # Carregar modelo
    model = HierarchicalMTL(X.shape[1], len(le1.classes_), len(le2.classes_)).to(device)
    model.load_state_dict(torch.load("results/best_hierarchical_model.pth"))
    model.eval()

    # Predição (usando apenas o set de validação para ser justo com o MCCV anterior)
    from sklearn.model_selection import train_test_split
    _, X_val, _, y1_val, _, y2_val = train_test_split(X, le1.transform(df["classe_nivel1"]), y2, test_size=0.2, random_state=42, stratify=y2)
    
    with torch.no_grad():
        out1, out2 = model(torch.FloatTensor(X_val).to(device))
        p1 = out1.argmax(1).cpu().numpy()
        p2 = out2.argmax(1).cpu().numpy()

    print("\n--- RESULTADO DEEP LEARNING (NÍVEL 2) ---")
    print(classification_report(y2_val, p2, target_names=le2.classes_))

    # Matriz de Confusão
    plt.figure(figsize=(12,10))
    cm = confusion_matrix(y2_val, p2)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le2.classes_, yticklabels=le2.classes_)
    plt.savefig("results/confusion_matrix_deep.png")
    print("✅ Matriz salva em results/confusion_matrix_deep.png")

if __name__ == "__main__":
    avaliar()