"""
Solución para caso 17: Predictor de eficiencia de combustible (MPG).

Función que procesa un dataset de automóviles, realiza ingeniería de variables
y entrena un modelo de Ridge Regression para predecir MPG.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def desarrollar_predictor_mpg(df_autos):
    """
    Procesa dataset de automóviles y entrena predictor Ridge para MPG.

    Pipeline:
    1. Limpia 'horsepower' (convierte '?' a NaN) y elimina valores nulos
    2. Crea variable de ingeniería: 'peso_por_caballo' = weight / horsepower
    3. One-Hot Encoding para 'origin'
    4. Divide datos: 75% entrenamiento, 25% prueba (random_state=42)
    5. Escala características con StandardScaler (solo en train)
    6. Entrena modelo Ridge Regression
    7. Retorna diccionario con: scaler_type, model_type, r2_score

    Parameters
    ----------
    df_autos : pd.DataFrame
        DataFrame con columnas: 'horsepower', 'weight', 'origin', 'mpg'
        La columna 'horsepower' puede contener strings '?'

    Returns
    -------
    dict
        Diccionario con las siguientes llaves exactas:
        - 'scaler_type': StandardScaler entrenado
        - 'model_type': Ridge Regression entrenado
        - 'r2_score': Coeficiente R^2 en datos de prueba
    """

    # Paso 1: Limpieza y transformación
    df = df_autos.copy()

    # Convertir 'horsepower' a float (? se convierte a NaN)
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

    # Eliminar filas con valores nulos
    df = df.dropna()

    # Crear nueva columna: 'peso_por_caballo'
    df["peso_por_caballo"] = df["weight"] / df["horsepower"]

    # Paso 2: One-Hot Encoding para 'origin'
    df = pd.get_dummies(df, columns=["origin"], drop_first=True)

    # Paso 3: Preparar características y objetivo
    X = df.drop("mpg", axis=1)
    y = df["mpg"]

    # Paso 4: División train-test (75-25 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Paso 5: Escalamiento (solo en train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Paso 6: Entrenar modelo Ridge
    modelo = Ridge()
    modelo.fit(X_train_scaled, y_train)

    # Paso 7: Calcular R2 en datos de prueba
    r2_score = modelo.score(X_test_scaled, y_test)

    # Retorno en formato diccionario (Nombres exactos exigidos por el validador)
    return {"scaler_type": scaler, "model_type": modelo, "r2_score": r2_score}


if __name__ == "__main__":
    print("=" * 70)
    print("EJEMPLO DE USO: desarrollar_predictor_mpg")
    print("=" * 70)

    # Crear dataset de ejemplo similar al de la pregunta
    np.random.seed(42)
    n_samples = 20

    datos = {
        "mpg": np.random.uniform(15, 45, n_samples),
        "horsepower": [
            100,
            150,
            "?",
            130,
            95,
            110,
            160,
            105,
            "?",
            120,
            85,
            90,
            140,
            115,
            100,
            105,
            125,
            150,
            95,
            110,
        ],
        "weight": np.random.uniform(2000, 5000, n_samples),
        "origin": np.random.choice(["USA", "Japan", "Europe"], n_samples),
    }

    df_ejemplo = pd.DataFrame(datos)

    print("\nDataFrame original (primeras 5 filas):")
    print(df_ejemplo.head())

    # Ejecutar función
    resultados = desarrollar_predictor_mpg(df_ejemplo)

    print("\nResultados:")
    print(f"  Scaler type: {type(resultados['scaler_type']).__name__}")
    print(f"  Modelo type: {type(resultados['model_type']).__name__}")
    print(f"  R2 Score: {resultados['r2_score']:.4f}")

    print("\nCaracterísticas del modelo:")
    print(f"  Coeficientes shape: {resultados['model_type'].coef_.shape}")
    print(f"  Intercept: {resultados['model_type'].intercept_:.4f}")
