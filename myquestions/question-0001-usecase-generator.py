import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_analisis_rendimiento_empleados():
    deptos = ['Ventas', 'IT', 'Marketing', 'Finanzas']
    num_records = random.randint(15, 30)
    
    data = []
    for i in range(num_records):
        dpto = random.choice(deptos) if random.random() > 0.1 else np.nan # 10% nulos
        horas = random.randint(100, 160) if random.random() > 0.15 else np.nan # 15% nulos
        proyectos = random.randint(1, 5)
        
        data.append({
            'empleado_id': f'EMP_{i}',
            'departamento': dpto,
            'horas_trabajadas': horas,
            'proyectos_completados': proyectos
        })
        
    df = pd.DataFrame(data)
    input_data = {'df': df.copy()}
    
    # Ground Truth
    df_expected = df.copy()
    df_expected = df_expected.dropna(subset=['departamento'])
    mediana_horas = df_expected['horas_trabajadas'].median()
    df_expected['horas_trabajadas'] = df_expected['horas_trabajadas'].fillna(mediana_horas)
    
    df_expected['eficiencia'] = df_expected['proyectos_completados'] / df_expected['horas_trabajadas']
    output_data = df_expected.groupby('departamento')[['eficiencia']].mean().sort_values(by='eficiencia', ascending=False)
    
    return input_data, output_data

    
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_analisis_rendimiento_empleados()
    
    print("\n=== INPUT (Diccionario) ===")
    print("DataFrame inicial (primeras filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (DataFrame procesado) ===")
    print(salida_esperada)