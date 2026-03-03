import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def generar_caso_de_uso_clasificacion_riesgo_credito():
    n_samples = random.randint(150, 250)
    n_features = random.randint(4, 8)
    
    X = np.random.randn(n_samples, n_features) * random.randint(1, 10)
    y = np.random.randint(0, 2, size=n_samples)
    
    test_size = round(random.uniform(0.2, 0.4), 2)
    rs = random.randint(1, 100)
    
    input_data = {
        'X': X,
        'y': y,
        'test_size': test_size,
        'random_state': rs
    }
    
    # Ground Truth
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel='rbf', random_state=rs)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    output_data = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='macro')
    }
    
    return input_data, output_data
  
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_clasificacion_riesgo_credito()
    
    print("\n=== INPUT (Diccionario) ===")
    print(f"Forma de X (filas, columnas): {entrada['X'].shape}")
    print(f"Forma de y (etiquetas): {entrada['y'].shape}")
    print(f"Test size: {entrada['test_size']}")
    print(f"Random state: {entrada['random_state']}")
    
    print("\n=== OUTPUT ESPERADO (Métricas) ===")
    print(salida_esperada)