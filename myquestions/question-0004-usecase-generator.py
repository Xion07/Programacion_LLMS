import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def generar_caso_de_uso_segmentacion_comportamental():
    n_rows = random.randint(100, 150)
    n_clusters = random.randint(2, 4)
    rs = random.randint(1, 42)
    
    data = {
        'tiempo_uso': np.random.uniform(10, 120, n_rows),
        'clics': np.random.randint(5, 50, n_rows),
        'compras': np.random.uniform(0, 500, n_rows),
        'categoria_favorita': random.choices(['Deportes', 'Tecnologia', 'Ropa'], k=n_rows) # Columna no numérica
    }
    df = pd.DataFrame(data)
    
    # Inyectar algunos NaNs en las columnas numéricas
    for col in ['tiempo_uso', 'clics', 'compras']:
        mask = np.random.choice([True, False], size=n_rows, p=[0.1, 0.9])
        df.loc[mask, col] = np.nan
        
    input_data = {
        'df': df.copy(),
        'n_clusters': n_clusters,
        'random_state': rs
    }
    
    # Ground Truth
    df_expected = df.copy()
    numerics = df_expected.select_dtypes(include=[np.number])
    
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(numerics)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=rs, n_init=10)
    kmeans.fit(scaled_data)
    etiquetas = kmeans.labels_
    
    output_data = etiquetas
    
    return input_data, output_data

   
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_segmentacion_comportamental()
    
    print("\n=== INPUT (Diccionario) ===")
    print(f"Número de clusters a buscar: {entrada['n_clusters']}")
    print(f"Random state: {entrada['random_state']}")
    print("DataFrame inicial (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (Etiquetas de los Clusters) ===")
    # Mostramos solo los primeros 30 para no saturar la terminal
    print(f"{salida_esperada[:30]} ... (mostrando los primeros 30 de {len(salida_esperada)})")