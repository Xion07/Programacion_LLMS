import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

def generar_caso_de_uso_comparar_arboles_regresion():
    n_samples = random.randint(200, 300)
    n_features = random.randint(5, 10)
    
    X = np.random.randn(n_samples, n_features)
    # Generar relación no lineal para que el árbol profundo a veces gane o pierda por overfitting
    y = X[:, 0]**2 + np.sin(X[:, 1]) * 10 + np.random.randn(n_samples) * 2
    
    cv_folds = random.randint(3, 5)
    
    input_data = {
        'X': X,
        'y': y,
        'cv_folds': cv_folds
    }
    
    # Ground Truth
    tree_shallow = DecisionTreeRegressor(max_depth=3)
    tree_deep = DecisionTreeRegressor(max_depth=10)
    
    scores_shallow = cross_val_score(tree_shallow, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    scores_deep = cross_val_score(tree_deep, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    
    rmse_shallow = np.mean(np.sqrt(-scores_shallow))
    rmse_deep = np.mean(np.sqrt(-scores_deep))
    
    mejor = "Superficial" if rmse_shallow < rmse_deep else "Profundo"
    
    output_data = {
        'rmse_superficial': rmse_shallow,
        'rmse_profundo': rmse_deep,
        'mejor_modelo': mejor
    }
    
    return input_data, output_data

   
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_comparar_arboles_regresion()
    
    print("\n=== INPUT (Diccionario) ===")
    print(f"Forma de X (filas, columnas): {entrada['X'].shape}")
    print(f"Forma de y (valores reales): {entrada['y'].shape}")
    print(f"Folds de validación cruzada (cv): {entrada['cv_folds']}")
    
    print("\n=== OUTPUT ESPERADO (Resultados de RMSE) ===")
    for clave, valor in salida_esperada.items():
        print(f"{clave}: {valor}")