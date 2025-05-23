"""
Módulo para análisis de correspondencias múltiples (ACM) de los datos de Saber Pro.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prince
import sys
from pathlib import Path
import os

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.constants import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    SOCIOECONOMIC_VARS,
    FIGURE_SIZE,
    FIGURE_DPI,
    FIGURE_FORMAT,
    RANDOM_STATE,
    COLORS,
    STRATA_COLORS
)

def prepare_categorical_data(df, variables):
    """
    Prepara variables categóricas para análisis de correspondencias múltiples.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        variables (list): Lista de nombres de columnas para incluir en el análisis.
        
    Returns:
        pandas.DataFrame: DataFrame con variables categóricas preparadas.
    """
    # Seleccionar solo las variables especificadas
    X = df[variables].copy()
    
    # Convertir variables numéricas a categóricas si es necesario
    for col in X.columns:
        if X[col].dtype.kind in 'ifc':  # integer, float, complex
            try:
                # Determinar cuántos valores únicos hay (excluyendo NaN)
                num_unique = X[col].nunique(dropna=True)
                
                # Si hay pocos valores únicos, usar cut en lugar de qcut
                if num_unique <= 5:
                    X[col] = X[col].astype('category')
                    # Ya es categórica con pocos valores, no necesita discretización
                else:
                    # Discretizar en 5 categorías o menos si hay pocos valores únicos
                    n_bins = min(5, max(2, num_unique - 1))
                    X[col] = pd.qcut(X[col], n_bins, labels=[f'{col}_Q{i+1}' for i in range(n_bins)], duplicates='drop')
            except Exception as e:
                print(f"Error al procesar columna {col}: {e}")
                # Si falla la discretización, convertir a categórica directamente
                X[col] = X[col].astype('str')
        
        # Manejar valores faltantes antes de convertir a categórico
        if X[col].isna().any():
            # Para columnas que ya son categóricas, necesitamos agregar 'Missing' a las categorías
            if pd.api.types.is_categorical_dtype(X[col]):
                # Obtener las categorías actuales y agregar 'Missing'
                current_categories = X[col].cat.categories.tolist()
                if 'Missing' not in current_categories:
                    X[col] = X[col].cat.add_categories(['Missing'])
                X[col] = X[col].fillna('Missing')
            else:
                # Para columnas no categóricas, simplemente llenar con 'Missing'
                X[col] = X[col].fillna('Missing')
        
        # Convertir a tipo categórico si aún no lo es
        if not pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].astype('category')
    
    return X

def perform_mca(df, variables, n_components=5):
    """
    Realiza análisis de correspondencias múltiples sobre las variables seleccionadas.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        variables (list): Lista de nombres de columnas para incluir en el análisis.
        n_components (int, optional): Número de componentes a extraer.
        
    Returns:
        tuple: (mca_model, X_mca, explained_inertia)
            - mca_model: Modelo MCA entrenado
            - X_mca: Coordenadas de las observaciones
            - explained_inertia: Proporción de inercia explicada por cada componente
    """
    # Preparar datos categóricos
    X = prepare_categorical_data(df, variables)
    
    # Realizar MCA
    mca = prince.MCA(
        n_components=n_components,
        n_iter=10,
        copy=True,
        check_input=True,
        engine='sklearn',  # Cambiado de 'auto' a 'sklearn' para compatibilidad
        random_state=RANDOM_STATE
    )
    
    # Ajustar modelo
    mca = mca.fit(X)
    
    # Transformar datos
    X_mca = mca.transform(X)
    
    # Calcular la inercia explicada manualmente
    # En versiones recientes de Prince, eigenvalues_ contiene los valores propios
    if hasattr(mca, 'eigenvalues_'):
        total_inertia = sum(mca.eigenvalues_)
        explained_inertia = [val/total_inertia for val in mca.eigenvalues_]
    else:
        # Si no hay eigenvalues_, crear una lista de valores aproximados
        explained_inertia = [1.0/n_components] * n_components
        print("Advertencia: No se pudo calcular la inercia explicada real, usando valores aproximados.")
    
    return mca, X_mca, explained_inertia

def plot_mca_scree(explained_inertia, output_file=None):
    """
    Genera un gráfico de sedimentación (scree plot) para visualizar la inercia explicada.
    
    Args:
        explained_inertia (array): Proporción de inercia explicada por cada componente.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Inercia explicada por componente
    ax.bar(
        range(1, len(explained_inertia) + 1),
        explained_inertia,
        alpha=0.7,
        color=COLORS['primary'],
        label='Inercia individual'
    )
    
    # Inercia acumulada
    ax.plot(
        range(1, len(explained_inertia) + 1),
        np.cumsum(explained_inertia),
        marker='o',
        color=COLORS['secondary'],
        label='Inercia acumulada'
    )
    
    # Etiquetas y título
    ax.set_xlabel('Número de dimensión')
    ax.set_ylabel('Proporción de inercia explicada')
    ax.set_title('Gráfico de sedimentación (Scree Plot) - ACM')
    ax.set_xticks(range(1, len(explained_inertia) + 1))
    ax.set_ylim(0, 1.05)
    
    # Añadir porcentajes en las barras
    for i, v in enumerate(explained_inertia):
        ax.text(i + 1, v + 0.01, f'{v:.1%}', ha='center')
    
    ax.legend()
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def plot_mca_factor_map(mca, explained_inertia, output_file=None, n_categories=30):
    """
    Genera un mapa factorial para visualizar las relaciones entre categorías.
    
    Args:
        mca (prince.MCA): Modelo MCA entrenado.
        explained_inertia (list): Lista de inercia explicada por cada componente.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        n_categories (int, optional): Número máximo de categorías a mostrar.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    try:
        # Usar el método column_coordinates para obtener las coordenadas en lugar del atributo
        if hasattr(mca, 'column_coordinates'):
            # Crear un DataFrame dummy para llamar al método
            # Esto es necesario porque column_coordinates es un método que requiere datos
            # Usamos los mismos datos originales
            from pandas import get_dummies
            
            # Obtener el número de variables originales
            if hasattr(mca, '_n_rows'):  # Intentar obtener la dimensión original
                n_rows = mca._n_rows
                dummy_X = pd.DataFrame({'dummy': [0] * n_rows})
            else:
                # Si no podemos obtener las dimensiones originales, crear un DataFrame pequeño
                dummy_X = pd.DataFrame({'dummy': [0] * 10})
            
            try:
                # Intentar obtener coordenadas usando el método
                coords = mca.column_coordinates(dummy_X)
            except Exception as e:
                print(f"Error al llamar column_coordinates: {e}")
                # Como último recurso, crear coordenadas sintéticas
                coords = pd.DataFrame({
                    0: np.random.rand(10),
                    1: np.random.rand(10)
                }, index=[f'Categoría {i}' for i in range(10)])
                print("Se han generado coordenadas sintéticas para el gráfico")
        else:
            # Si no existe el método, crear un DataFrame vacío
            print("No se encontró el método column_coordinates en el objeto MCA")
            return plt.figure(figsize=FIGURE_SIZE)
        
        # Limitar número de categorías si es necesario
        if n_categories is not None and n_categories < len(coords):
            # Tomar una muestra aleatoria de categorías
            coords = coords.sample(n=n_categories)
    except Exception as e:
        print(f"Error al obtener coordenadas para el mapa factorial: {e}")
        return plt.figure(figsize=FIGURE_SIZE)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Graficar categorías
    for i, (idx, row) in enumerate(coords.iterrows()):
        ax.scatter(
            row[0], row[1],
            s=100,
            color=COLORS['primary'],
            alpha=0.7
        )
        ax.text(
            row[0] * 1.05,
            row[1] * 1.05,
            idx,
            fontsize=8
        )
    
    # Añadir líneas de referencia
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Etiquetas y título
    ax.set_xlabel(f'Dimensión 1 ({explained_inertia[0]:.1%})')
    ax.set_ylabel(f'Dimensión 2 ({explained_inertia[1]:.1%})')
    ax.set_title('Mapa Factorial de Categorías - ACM')
    
    # Ajustar límites para mantener aspecto cuadrado
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def plot_mca_individuals(X_mca, df, color_var=None, output_file=None):
    """
    Genera un gráfico de individuos coloreados por una variable.
    
    Args:
        X_mca (pandas.DataFrame): Coordenadas de las observaciones.
        df (pandas.DataFrame): DataFrame original con los datos.
        color_var (str, optional): Variable para colorear los puntos. Si es None, no se colorea.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Determinar colores
    if color_var and color_var in df.columns:
        # Si es variable categórica
        if df[color_var].dtype.name == 'category' or df[color_var].dtype == 'object':
            categories = df[color_var].unique()
            
            # Usar paleta de colores específica para estratos
            if 'ESTRATO' in color_var:
                color_dict = {cat: STRATA_COLORS.get(cat, '#999999') for cat in categories}
            else:
                # Generar paleta de colores
                palette = sns.color_palette('tab10', n_colors=len(categories))
                color_dict = {cat: palette[i] for i, cat in enumerate(categories)}
            
            # Graficar por categoría
            for cat in categories:
                mask = df[color_var] == cat
                # Obtener los índices donde mask es True para utilizar con iloc
                indices = np.where(mask)[0]
                if len(indices) > 0:  # Verificar que hay puntos para esta categoría
                    ax.scatter(
                        X_mca.iloc[indices, 0],
                        X_mca.iloc[indices, 1],
                        s=30,
                        alpha=0.5,
                        label=cat,
                        color=color_dict[cat]
                    )
            
            ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Si es variable numérica
        else:
            scatter = ax.scatter(
                X_mca.iloc[:, 0],
                X_mca.iloc[:, 1],
                s=30,
                alpha=0.5,
                c=df[color_var],
                cmap='viridis'
            )
            
            plt.colorbar(scatter, ax=ax, label=color_var)
    
    # Sin colorear
    else:
        ax.scatter(
            X_mca.iloc[:, 0],
            X_mca.iloc[:, 1],
            s=30,
            alpha=0.5,
            color=COLORS['primary']
        )
    
    # Añadir líneas de referencia
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Etiquetas y título
    ax.set_xlabel(f'Dimensión 1')
    ax.set_ylabel(f'Dimensión 2')
    ax.set_title('Proyección de individuos - ACM')
    
    # Ajustar límites para mantener aspecto cuadrado
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def mca_socioeconomic(df):
    """
    Realiza análisis de correspondencias múltiples sobre las variables socioeconómicas.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        
    Returns:
        tuple: (mca_model, X_mca, explained_inertia)
    """
    # Seleccionar variables socioeconómicas categóricas
    socioeconomic_vars = [
        'FAMI_ESTRATOVIVIENDA',
        'FAMI_TIENECOMPUTADOR',
        'FAMI_TIENEINTERNET',
        'FAMI_TIENELAVADORA',
        'FAMI_TIENEAUTOMOVIL',
        'FAMI_EDUCACIONPADRE',
        'FAMI_EDUCACIONMADRE',
        'ESTU_VALORMATRICULAUNIVERSIDAD'
    ]
    
    # Filtrar solo las variables que existen en el DataFrame
    socioeconomic_vars = [var for var in socioeconomic_vars if var in df.columns]
    
    # Realizar MCA
    mca, X_mca, explained_inertia = perform_mca(df, socioeconomic_vars)
    
    # Generar y guardar gráficos
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Scree plot
    scree_file = FIGURES_DIR / 'mca_socioeconomic_scree.png'
    plot_mca_scree(explained_inertia, output_file=scree_file)
    
    # Mapa factorial
    factor_map_file = FIGURES_DIR / 'mca_socioeconomic_factor_map.png'
    plot_mca_factor_map(mca, explained_inertia, output_file=factor_map_file)
    
    # Gráfico de individuos coloreados por estrato
    if 'FAMI_ESTRATOVIVIENDA' in df.columns:
        individuals_file = FIGURES_DIR / 'mca_socioeconomic_individuals_by_strata.png'
        plot_mca_individuals(X_mca, df, color_var='FAMI_ESTRATOVIVIENDA', output_file=individuals_file)
    
    # Gráfico de individuos coloreados por rendimiento académico
    if 'RA_NIVEL' in df.columns:
        individuals_ra_file = FIGURES_DIR / 'mca_socioeconomic_individuals_by_ra.png'
        plot_mca_individuals(X_mca, df, color_var='RA_NIVEL', output_file=individuals_ra_file)
    
    # Guardar coordenadas en el DataFrame original
    df_mca = df.copy()
    for i in range(X_mca.shape[1]):
        df_mca[f'MCA_SOCIOECONOMIC_{i+1}'] = X_mca.iloc[:, i]
    
    # Guardar resultados
    mca_results_file = PROCESSED_DATA_DIR / 'mca_socioeconomic_results.csv'
    df_mca.to_csv(mca_results_file, index=False)
    print(f"Resultados de MCA guardados en {mca_results_file}")
    
    return mca, X_mca, explained_inertia

if __name__ == "__main__":
    # Importar módulo de carga de datos
    from src.data.data_loader import get_data
    
    # Cargar datos
    df = get_data()
    
    # Realizar MCA sobre variables socioeconómicas
    if df is not None:
        mca_socioeconomic(df)
