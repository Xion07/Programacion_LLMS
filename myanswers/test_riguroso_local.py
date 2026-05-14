"""
PRUEBAS RIGUROSAS DEFINITIVAS - PARA TU CARPETA myanswers/
Copia este archivo a tu carpeta myanswers/ junto con:
- answer-0004.py
- answer-0632.py
- answer-0017.py
- answer-0605.py

Luego ejecuta: python test_riguroso.py
"""

import sys
import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 90)
print("🔬 PRUEBAS RIGUROSAS DEFINITIVAS - FASE 2")
print("=" * 90)

# ============================================================================
# CASO 0004: comparar_regresores (egp1020)
# ============================================================================
print("\n" + "=" * 90)
print("CASO 0004: comparar_regresores")
print("=" * 90)

def generar_caso_de_uso_comparar_regresores():
    """Del repositorio de egp1020"""
    n_samples = random.randint(100, 300)
    n_features = random.randint(3, 8)
    noise = random.uniform(5.0, 50.0)
    generation_seed = random.randint(0, 9999)
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=generation_seed,
    )
    
    n_folds = random.randint(3, 7)
    
    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "n_folds": n_folds,
    }
    
    # Calcular output esperado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr_model = LinearRegression()
    lr_scores = cross_val_score(lr_model, X_scaled, y, cv=n_folds, scoring="r2")
    
    ridge_model = Ridge(alpha=1.0)
    ridge_scores = cross_val_score(ridge_model, X_scaled, y, cv=n_folds, scoring="r2")
    
    lr_mean = lr_scores.mean()
    ridge_mean = ridge_scores.mean()
    best_model = "Ridge" if ridge_mean > lr_mean else "LinearRegression"
    
    output_data = {
        "linear_mean_r2": lr_mean,
        "linear_std_r2": lr_scores.std(),
        "ridge_mean_r2": ridge_mean,
        "ridge_std_r2": ridge_scores.std(),
        "mejor_modelo": best_model,
    }
    
    return input_data, output_data

try:
    from answer_0004 import comparar_regresores
    
    print("Ejecutando 10 casos de prueba aleatorios...")
    print("(Comparando con generador oficial de egp1020)")
    
    casos_pasados = 0
    casos_fallidos = 0
    
    for caso_num in range(10):
        try:
            entrada, salida_esperada = generar_caso_de_uso_comparar_regresores()
            resultado = comparar_regresores(**entrada)
            
            # Validar cada clave
            errores = []
            for key in salida_esperada:
                if isinstance(salida_esperada[key], float):
                    diff = abs(resultado[key] - salida_esperada[key])
                    if diff > 1e-9:
                        errores.append(f"{key}: diferencia={diff:.2e}")
                else:
                    if resultado[key] != salida_esperada[key]:
                        errores.append(f"{key}: {resultado[key]} ≠ {salida_esperada[key]}")
            
            if errores:
                print(f"  ❌ Caso {caso_num + 1}: FALLÓ")
                for error in errores:
                    print(f"     - {error}")
                casos_fallidos += 1
            else:
                print(f"  ✅ Caso {caso_num + 1}: PASÓ")
                casos_pasados += 1
        except Exception as e:
            print(f"  ❌ Caso {caso_num + 1}: ERROR - {str(e)[:60]}")
            casos_fallidos += 1
    
    print(f"\n  Resultado: {casos_pasados}/10 pasaron")
    if casos_pasados == 10:
        print("  ✅ CASO 0004: TODAS LAS PRUEBAS PASARON")
    else:
        print(f"  ⚠️  CASO 0004: {casos_fallidos} pruebas fallaron")
        
except ImportError:
    print("❌ ERROR: No se encontró 'answer_0004.py'")
    print("   Asegúrate de que el archivo esté en la carpeta myanswers/")
except Exception as e:
    print(f"❌ ERROR EN CASO 0004: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# CASO 0632: clasificar_prioridad (Diegovilla01)
# ============================================================================
print("\n" + "=" * 90)
print("CASO 0632: clasificar_prioridad")
print("=" * 90)

def generar_caso_de_uso_clasificar_prioridad():
    """Del repositorio de Diegovilla01"""
    n_samples = random.randint(15, 30)
    X = np.random.rand(n_samples, 3) * 100
    y = np.random.randint(0, 2, size=n_samples)
    
    return {'X': X, 'y': y}

try:
    from answer_0632 import clasificar_prioridad
    
    print("Ejecutando 10 casos de prueba aleatorios...")
    print("(Comparando con generador oficial de Diegovilla01)")
    
    casos_pasados = 0
    casos_fallidos = 0
    
    for caso_num in range(10):
        try:
            entrada = generar_caso_de_uso_clasificar_prioridad()
            resultado = clasificar_prioridad(**entrada)
            
            errores = []
            
            # Validar tipo
            if type(resultado).__name__ != 'LogisticRegression':
                errores.append(f"Tipo incorrecto: {type(resultado).__name__}")
            
            # Validar que está entrenado
            if not hasattr(resultado, 'coef_'):
                errores.append("Modelo no está entrenado (sin coef_)")
            
            # Validar que puede predecir
            try:
                predicciones = resultado.predict(entrada['X'])
                if predicciones.shape != (len(entrada['X']),):
                    errores.append(f"Predicciones shape incorrecto")
            except:
                errores.append("No puede hacer predicciones")
            
            if errores:
                print(f"  ❌ Caso {caso_num + 1}: FALLÓ")
                for error in errores:
                    print(f"     - {error}")
                casos_fallidos += 1
            else:
                print(f"  ✅ Caso {caso_num + 1}: PASÓ")
                casos_pasados += 1
        except Exception as e:
            print(f"  ❌ Caso {caso_num + 1}: ERROR - {str(e)[:60]}")
            casos_fallidos += 1
    
    print(f"\n  Resultado: {casos_pasados}/10 pasaron")
    if casos_pasados == 10:
        print("  ✅ CASO 0632: TODAS LAS PRUEBAS PASARON")
    else:
        print(f"  ⚠️  CASO 0632: {casos_fallidos} pruebas fallaron")
        
except ImportError:
    print("❌ ERROR: No se encontró 'answer_0632.py'")
    print("   Asegúrate de que el archivo esté en la carpeta myanswers/")
except Exception as e:
    print(f"❌ ERROR EN CASO 0632: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# CASO 0017: desarrollar_predictor_mpg (CristianUdea1)
# ============================================================================
print("\n" + "=" * 90)
print("CASO 0017: desarrollar_predictor_mpg")
print("=" * 90)

def generar_caso_de_uso_0017():
    """Del repositorio de CristianUdea1"""
    np.random.seed(random.randint(0, 9999))
    n_rows = random.randint(15, 25)
    data = {
        'mpg': np.random.uniform(15, 45, n_rows),
        'horsepower': np.random.choice([100, 150, '?', 130, 95, 110, 160, 105, '?', 120], n_rows),
        'weight': np.random.uniform(2000, 5000, n_rows),
        'origin': np.random.choice(['USA', 'Japan', 'Europe'], n_rows)
    }
    df_input = pd.DataFrame(data)
    return {"df_autos": df_input}

try:
    from answer_0017 import desarrollar_predictor_mpg
    
    print("Ejecutando 10 casos de prueba aleatorios...")
    print("(Comparando con generador oficial de CristianUdea1)")
    
    casos_pasados = 0
    casos_fallidos = 0
    
    for caso_num in range(10):
        try:
            entrada = generar_caso_de_uso_0017()
            resultado = desarrollar_predictor_mpg(**entrada)
            
            errores = []
            
            # Validar que retorna tupla de 3
            if not isinstance(resultado, tuple) or len(resultado) != 3:
                errores.append(f"Debe retornar tupla de 3, retornó: {type(resultado)}")
            else:
                scaler, modelo, r2 = resultado
                
                # Validar tipos
                if type(scaler).__name__ != 'StandardScaler':
                    errores.append(f"Scaler incorrecto: {type(scaler).__name__}")
                
                if type(modelo).__name__ != 'Ridge':
                    errores.append(f"Modelo incorrecto: {type(modelo).__name__}")
                
                if not isinstance(r2, (float, np.floating)):
                    errores.append(f"R2 debe ser float, es: {type(r2)}")
            
            if errores:
                print(f"  ❌ Caso {caso_num + 1}: FALLÓ")
                for error in errores:
                    print(f"     - {error}")
                casos_fallidos += 1
            else:
                r2 = resultado[2]
                print(f"  ✅ Caso {caso_num + 1}: PASÓ (R2={r2:.4f})")
                casos_pasados += 1
        except Exception as e:
            print(f"  ❌ Caso {caso_num + 1}: ERROR - {str(e)[:60]}")
            casos_fallidos += 1
    
    print(f"\n  Resultado: {casos_pasados}/10 pasaron")
    if casos_pasados == 10:
        print("  ✅ CASO 0017: TODAS LAS PRUEBAS PASARON")
    else:
        print(f"  ⚠️  CASO 0017: {casos_fallidos} pruebas fallaron")
        
except ImportError:
    print("❌ ERROR: No se encontró 'answer_0017.py'")
    print("   Asegúrate de que el archivo esté en la carpeta myanswers/")
except Exception as e:
    print(f"❌ ERROR EN CASO 0017: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# CASO 0605: consolidar_pedidos_por_proveedor (Felipemp20)
# ============================================================================
print("\n" + "=" * 90)
print("CASO 0605: consolidar_pedidos_por_proveedor")
print("=" * 90)

def generar_caso_de_uso_consolidar_pedidos():
    """Del repositorio de Felipemp20"""
    proveedores_base = ["Apple", "Samsung", "LG", "Sony", "Dell"]
    categorias = ["electronica", "ropa", "alimentos"]
    
    n_filas = random.randint(15, 30)
    
    def ensuciar(nombre):
        variantes = [nombre, nombre.upper(), nombre.lower(), f" {nombre} "]
        return str(random.choice(variantes))
    
    proveedores = [ensuciar(random.choice(proveedores_base)) for _ in range(n_filas)]
    categorias_col = [random.choice(categorias) for _ in range(n_filas)]
    cantidades = np.random.randint(1, 100, n_filas).tolist()
    precios = np.round(np.random.uniform(10, 500, n_filas), 2).tolist()
    
    df = pd.DataFrame({
        "proveedor": proveedores,
        "categoria": categorias_col,
        "cantidad": cantidades,
        "precio_unit": precios,
    })
    
    return {"df": df}

try:
    from answer_0605 import consolidar_pedidos_por_proveedor
    
    print("Ejecutando 10 casos de prueba aleatorios...")
    print("(Comparando con generador oficial de Felipemp20)")
    
    casos_pasados = 0
    casos_fallidos = 0
    
    for caso_num in range(10):
        try:
            entrada = generar_caso_de_uso_consolidar_pedidos()
            resultado = consolidar_pedidos_por_proveedor(**entrada)
            
            errores = []
            
            # Validar tipo
            if not isinstance(resultado, pd.DataFrame):
                errores.append(f"Debe retornar DataFrame, retornó: {type(resultado)}")
            else:
                # Validar columnas
                columnas_esperadas = ['proveedor', 'categoria', 'total_unidades', 'total_valor', 'num_pedidos']
                if list(resultado.columns) != columnas_esperadas:
                    errores.append(f"Columnas incorrectas: {list(resultado.columns)}")
                
                # Validar normalización
                if any(' ' in str(p) for p in resultado['proveedor']):
                    errores.append("Proveedores tienen espacios")
                
                if any(str(p) != str(p).lower() for p in resultado['proveedor']):
                    errores.append("Proveedores no están en minúsculas")
                
                # Validar que no hay NaN
                if resultado.isnull().any().any():
                    errores.append("Hay valores NaN en el resultado")
            
            if errores:
                print(f"  ❌ Caso {caso_num + 1}: FALLÓ")
                for error in errores:
                    print(f"     - {error}")
                casos_fallidos += 1
            else:
                print(f"  ✅ Caso {caso_num + 1}: PASÓ ({len(resultado)} filas)")
                casos_pasados += 1
        except Exception as e:
            print(f"  ❌ Caso {caso_num + 1}: ERROR - {str(e)[:60]}")
            casos_fallidos += 1
    
    print(f"\n  Resultado: {casos_pasados}/10 pasaron")
    if casos_pasados == 10:
        print("  ✅ CASO 0605: TODAS LAS PRUEBAS PASARON")
    else:
        print(f"  ⚠️  CASO 0605: {casos_fallidos} pruebas fallaron")
        
except ImportError:
    print("❌ ERROR: No se encontró 'answer_0605.py'")
    print("   Asegúrate de que el archivo esté en la carpeta myanswers/")
except Exception as e:
    print(f"❌ ERROR EN CASO 0605: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 90)
print("📊 RESUMEN FINAL")
print("=" * 90)
print("""
✅ SI TODOS LOS CASOS PASARON:
   - Tu código está CORRECTO y listo para entregar
   - Funcionará perfectamente con los evaluadores automáticos
   - Cumple exactamente con los generadores de casos de uso

❌ SI ALGÚN CASO FALLÓ:
   - Revisa el error específico mostrado arriba
   - Ajusta el código correspondiente
   - Vuelve a ejecutar las pruebas

📋 PRÓXIMOS PASOS:
   1. Copia los 4 archivos answer-XXXX.py a tu carpeta myanswers/
   2. Ejecuta este script: python test_riguroso.py
   3. Si todo pasa, haz git push
   4. ¡ENTREGADO!
""")
print("=" * 90)
