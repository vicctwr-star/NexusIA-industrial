import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# 1. CRIAÇÃO DO DATASET ARTIFICIAL
# ============================================================

np.random.seed(42)

n = 5000

temperatura = np.random.normal(60, 15, n)
vibracao = np.random.normal(0.6, 0.35, n)
pressao = np.random.normal(4.0, 0.5, n)
horas_uso = np.random.uniform(100, 1000, n)
energia = np.random.normal(20, 5, n)


# Limitar valores para ficarem mais realistas
temperatura = np.clip(temperatura, 20, 110)
vibracao = np.clip(vibracao, 0, 2)
pressao = np.clip(pressao, 2, 5)
energia = np.clip(energia, 5, 40)


# ============================================================
# 2. CRIAÇÃO DA VARIÁVEL DE FALHA
# ============================================================

risco = (
    (temperatura > 75).astype(int)
    + (vibracao > 0.9).astype(int)
    + (pressao < 3.6).astype(int)
    + (horas_uso > 750).astype(int)
    + (energia > 27).astype(int)
)

falha = (risco >= 3).astype(int)


# ============================================================
# 3. DATAFRAME
# ============================================================

df = pd.DataFrame({
    "temperatura": temperatura,
    "vibracao": vibracao,
    "pressao": pressao,
    "horas_uso": horas_uso,
    "energia": energia,
    "falha": falha
})


# ============================================================
# 4. TREINAMENTO
# ============================================================

X = df[
    [
        "temperatura",
        "vibracao",
        "pressao",
        "horas_uso",
        "energia"
    ]
]

y = df["falha"]


X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modelo.fit(X_treino, y_treino)


# ============================================================
# 5. PRECISÃO
# ============================================================

previsoes = modelo.predict(X_teste)

precisao = accuracy_score(
    y_teste,
    previsoes
)


# ============================================================
# 6. FUNÇÃO PARA ANALISAR UMA MÁQUINA
# ============================================================

def analisar_maquina(
    temperatura,
    vibracao,
    pressao,
    horas_uso,
    energia
):

    dados = pd.DataFrame({
        "temperatura": [temperatura],
        "vibracao": [vibracao],
        "pressao": [pressao],
        "horas_uso": [horas_uso],
        "energia": [energia]
    })

    risco = modelo.predict_proba(dados)[0][1] * 100

    if risco < 30:
        status = "🟢 NORMAL"
        recomendacao = "Máquina operando normalmente."

    elif risco < 70:
        status = "🟡 ATENÇÃO"
        recomendacao = "Recomenda-se realizar uma inspeção."

    else:
        status = "🔴 CRÍTICO"
        recomendacao = "Realizar manutenção preventiva."

    return risco, status, recomendacao