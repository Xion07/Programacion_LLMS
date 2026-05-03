"""
Solución para la pregunta 0004: Comparación de modelos de regresión.

Función que compara LinearRegression y Ridge usando validación cruzada
con escalado de características.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score


def comparar_regresores(X, y, n_folds):
    """
    Compara el rendimiento de LinearRegression y Ridge usando cross-validation.
    
    Pipeline:
    1. Escala las características con StandardScaler
    2. Evalúa LinearRegression con cross_val_score (scoring="r2")
    3. Evalúa Ridge(alpha=1.0) con cross_val_score (scoring="r2")
    4. Determina cuál modelo tiene mejor rendimiento promedio
    
    Parameters
    ----------
    X : np.ndarray
        Matriz de características de forma (n_samples, n_features)
    y : np.ndarray
        Vector objetivo de forma (n_samples,)
    n_folds : int
        Número de folds para validación cruzada (entre 3 y 10)
    
    Returns
    -------
    dict
        Diccionario con las claves:
        - "linear_mean_r2" (float): Media del R2 de LinearRegression
        - "linear_std_r2" (float): Desviación estándar del R2 de LinearRegression
        - "ridge_mean_r2" (float): Media del R2 de Ridge
        - "ridge_std_r2" (float): Desviación estándar del R2 de Ridge
        - "mejor_modelo" (str): "LinearRegression" o "Ridge"
    
    Example
    -------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    >>> resultado = comparar_regresores(X, y, n_folds=5)
    >>> print(resultado)
    {'linear_mean_r2': 0.9635..., 'linear_std_r2': 0.0118..., 
     'ridge_mean_r2': 0.9632..., 'ridge_std_r2': 0.0117..., 
     'mejor_modelo': 'LinearRegression'}
    """
    
    # Paso 1: Escalar las características con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Paso 2: Evaluar LinearRegression
    lr_model = LinearRegression()
    lr_scores = cross_val_score(
        lr_model, X_scaled, y, 
        cv=n_folds, 
        scoring="r2"
    )
    
    # Paso 3: Evaluar Ridge con alpha=1.0
    ridge_model = Ridge(alpha=1.0)
    ridge_scores = cross_val_score(
        ridge_model, X_scaled, y,
        cv=n_folds,
        scoring="r2"
    )
    
    # Paso 4: Determinar cuál modelo tiene mejor rendimiento promedio
    lr_mean = lr_scores.mean()
    ridge_mean = ridge_scores.mean()
    
    # Si ambos tienen exactamente el mismo mean R2, retornar LinearRegression
    if ridge_mean > lr_mean:
        mejor_modelo = "Ridge"
    else:
        mejor_modelo = "LinearRegression"
    
    # Retornar diccionario con resultados
    return {
        "linear_mean_r2": lr_mean,
        "linear_std_r2": lr_scores.std(),
        "ridge_mean_r2": ridge_mean,
        "ridge_std_r2": ridge_scores.std(),
        "mejor_modelo": mejor_modelo,
    }


if __name__ == "__main__":
    # Ejemplo de uso básico
    from sklearn.datasets import make_regression
    
    print("=" * 60)
    print("EJEMPLO DE USO BÁSICO")
    print("=" * 60)
    
    X, y = make_regression(
        n_samples=200, 
        n_features=5, 
        noise=10, 
        random_state=42
    )
    
    resultado = comparar_regresores(X, y, n_folds=5)
    
    print("\nResultado de la comparación:")
    print(f"  Linear Mean R2:  {resultado['linear_mean_r2']:.6f}")
    print(f"  Linear Std R2:   {resultado['linear_std_r2']:.6f}")
    print(f"  Ridge Mean R2:   {resultado['ridge_mean_r2']:.6f}")
    print(f"  Ridge Std R2:    {resultado['ridge_std_r2']:.6f}")
    print(f"  Mejor Modelo:    {resultado['mejor_modelo']}")
