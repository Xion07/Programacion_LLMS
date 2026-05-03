"""
Solución para el caso 632: Clasificación de prioridad de envío logístico.

Función que escala características y entrena un modelo LogisticRegression
para clasificar la prioridad de envío (Estándar vs Urgente).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def clasificar_prioridad(X, y):
    """
    Entrena un modelo de LogisticRegression para clasificar prioridad de envíos.
    
    Pipeline:
    1. Escala las características X usando StandardScaler
    2. Entrena un modelo LogisticRegression con los datos escalados
    3. Retorna el modelo entrenado
    
    Parameters
    ----------
    X : np.ndarray or array-like
        Características de los pedidos (peso, distancia, costo)
        Forma: (n_samples, n_features)
    y : np.ndarray or array-like
        Etiquetas de prioridad (0: Estándar, 1: Urgente)
        Forma: (n_samples,)
    
    Returns
    -------
    LogisticRegression
        Modelo de LogisticRegression entrenado sobre datos escalados
    
    Example
    -------
    >>> import numpy as np
    >>> X = np.random.rand(20, 3) * 100
    >>> y = np.random.randint(0, 2, size=20)
    >>> modelo = clasificar_prioridad(X, y)
    >>> print(type(modelo))
    <class 'sklearn.linear_model._logistic.LogisticRegression'>
    """
    
    # Paso 1: Escalar las características con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Paso 2: Entrenar modelo LogisticRegression
    modelo = LogisticRegression()
    modelo.fit(X_scaled, y)
    
    # Paso 3: Retornar el modelo entrenado
    return modelo


if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("EJEMPLO DE USO: clasificar_prioridad")
    print("=" * 60)
    
    # Crear datos de ejemplo
    np.random.seed(42)
    X_ejemplo = np.random.rand(20, 3) * 100  # 20 muestras, 3 características
    y_ejemplo = np.random.randint(0, 2, size=20)  # Etiquetas binarias
    
    print("\nDatos de entrada:")
    print(f"  X shape: {X_ejemplo.shape}")
    print(f"  y shape: {y_ejemplo.shape}")
    print(f"  Clases: {np.unique(y_ejemplo)}")
    
    # Entrenar modelo
    modelo_entrenado = clasificar_prioridad(X_ejemplo, y_ejemplo)
    
    print("\nModelo entrenado:")
    print(f"  Tipo: {type(modelo_entrenado)}")
    print(f"  Coeficientes shape: {modelo_entrenado.coef_.shape}")
    print(f"  Intercept: {modelo_entrenado.intercept_}")
    
    # Hacer predicciones de ejemplo
    predicciones = modelo_entrenado.predict(X_ejemplo)
    print(f"\nPredicciones en datos de entrenamiento:")
    print(f"  Primeras 5 predicciones: {predicciones[:5]}")
    print(f"  Distribución: {np.unique(predicciones, return_counts=True)}")
