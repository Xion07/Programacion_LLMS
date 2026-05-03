"""
Solución para caso 605: Consolidación de pedidos por proveedor.

Función que normaliza datos de proveedores, calcula valores totales de pedidos
y agrupa la información por proveedor y categoría.
"""

import pandas as pd
import numpy as np


def consolidar_pedidos_por_proveedor(df):
    """
    Consolida pedidos de múltiples proveedores normalizando datos y agrupando.
    
    Pipeline:
    1. Normaliza la columna 'proveedor': elimina espacios y convierte a minúsculas
    2. Calcula 'valor_total' = cantidad * precio_unit para cada pedido
    3. Agrupa por 'proveedor' y 'categoria' calculando:
       - total_unidades: suma de cantidad
       - total_valor: suma de valor_total
       - num_pedidos: número de filas en el grupo
    4. Ordena por proveedor (ascendente) y luego por total_valor (descendente)
    5. Reinicia el índice
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas: 'proveedor', 'categoria', 'cantidad', 'precio_unit'
        La columna 'proveedor' puede contener espacios extras y variaciones de mayúsculas
    
    Returns
    -------
    pd.DataFrame
        DataFrame consolidado con columnas:
        ['proveedor', 'categoria', 'total_unidades', 'total_valor', 'num_pedidos']
    
    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'proveedor': [' Apple ', 'apple', 'APPLE'],
    ...     'categoria': ['electronica', 'electronica', 'ropa'],
    ...     'cantidad': [10, 20, 15],
    ...     'precio_unit': [100.0, 100.0, 50.0]
    ... })
    >>> resultado = consolidar_pedidos_por_proveedor(df)
    >>> print(resultado)
    """
    
    # Crear copia para no modificar el dataframe original
    df_trabajo = df.copy()
    
    # Paso 1: Normalizar la columna 'proveedor'
    # Eliminar espacios al inicio/fin y convertir a minúsculas
    df_trabajo['proveedor'] = df_trabajo['proveedor'].str.strip().str.lower()
    
    # Paso 2: Calcular valor_total para cada pedido
    df_trabajo['valor_total'] = df_trabajo['cantidad'] * df_trabajo['precio_unit']
    
    # Paso 3: Agrupar por proveedor y categoria
    resultado = (
        df_trabajo.groupby(['proveedor', 'categoria'], as_index=False)
        .agg(
            total_unidades=('cantidad', 'sum'),
            total_valor=('valor_total', 'sum'),
            num_pedidos=('cantidad', 'count'),
        )
    )
    
    # Paso 4: Ordenar por proveedor (ascendente) y luego por total_valor (descendente)
    resultado = resultado.sort_values(
        ['proveedor', 'total_valor'],
        ascending=[True, False]
    )
    
    # Paso 5: Reiniciar índice
    resultado = resultado.reset_index(drop=True)
    
    return resultado


if __name__ == "__main__":
    print("=" * 70)
    print("EJEMPLO DE USO: consolidar_pedidos_por_proveedor")
    print("=" * 70)
    
    # Crear dataset de ejemplo con datos "sucios"
    datos_ejemplo = {
        'proveedor': [' Apple ', 'apple', 'APPLE', 'Samsung', ' SAMSUNG ',
                      'LG', 'lg', 'Apple', 'Samsung', 'LG'],
        'categoria': ['electronica', 'electronica', 'ropa', 'electronica',
                      'ropa', 'electronica', 'ropa', 'electronica', 'ropa', 'electronica'],
        'cantidad': [10, 20, 15, 25, 30, 12, 18, 22, 14, 16],
        'precio_unit': [100.0, 100.0, 50.0, 200.0, 75.0, 150.0, 60.0, 100.0, 75.0, 150.0]
    }
    
    df_ejemplo = pd.DataFrame(datos_ejemplo)
    
    print("\nDataFrame original (sin normalizar):")
    print(df_ejemplo)
    
    print("\n" + "-" * 70)
    print("Ejecutando consolidar_pedidos_por_proveedor...")
    print("-" * 70)
    
    # Ejecutar función
    resultado_consolidado = consolidar_pedidos_por_proveedor(df_ejemplo)
    
    print("\nDataFrame consolidado:")
    print(resultado_consolidado)
    
    print("\nCaracterísticas del resultado:")
    print(f"  Número de grupos únicos: {len(resultado_consolidado)}")
    print(f"  Columnas: {list(resultado_consolidado.columns)}")
    print(f"  Tipo de dato total_valor: {resultado_consolidado['total_valor'].dtype}")
    
    print("\nResumen por proveedor:")
    print(resultado_consolidado.groupby('proveedor')[['total_unidades', 'total_valor']].sum())
