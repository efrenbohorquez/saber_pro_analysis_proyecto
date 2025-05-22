"""
Módulo para análisis de componentes principales (ACP) de los datos de Saber Pro.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.constants import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    ACADEMIC_VARS,
    FIGURE_SIZE,
    FIGURE_DPI,
    FIGURE_FORMAT,
    N_COMPONENTS_PCA,
    RANDOM_STATE,
    COLORS
)

def perform_pca(df, variables, n_components=None, scale=True):
    """
    Realiza análisis de componentes principales sobre las variables seleccionadas.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        variables (list): Lista de nombres de columnas para incluir en el análisis.
        n_components (int, optional): Número de componentes a extraer. Si es None, se determina automáticamente.
        scale (bool, optional): Si es True, estandariza las variables antes del análisis.
        
    Returns:
        tuple: (pca_model, X_pca, explained_variance_ratio, loadings)
            - pca_model: Modelo PCA entrenado
            - X_pca: Datos transformados
            - explained_variance_ratio: Proporción de varianza explicada por cada componente
            - loadings: Cargas factoriales de cada variable en cada componente
    """
    # Seleccionar solo las variables numéricas
    X = df[variables].select_dtypes(include=['number'])
    
    # Manejar valores faltantes
    X = X.fillna(X.mean())
    
    # Estandarizar si es necesario
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Determinar número de componentes si no se especifica
    if n_components is None:
        n_components = min(len(X.columns), N_COMPONENTS_PCA)
    
    # Realizar PCA
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calcular cargas factoriales
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return pca, X_pca, pca.explained_variance_ratio_, loadings

def plot_scree(explained_variance_ratio, output_file=None):
    """
    Genera un gráfico de sedimentación (scree plot) para visualizar la varianza explicada.
    
    Args:
        explained_variance_ratio (array): Proporción de varianza explicada por cada componente.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Varianza explicada por componente
    ax.bar(
        range(1, len(explained_variance_ratio) + 1),
        explained_variance_ratio,
        alpha=0.7,
        color=COLORS['primary'],
        label='Varianza individual'
    )
    
    # Varianza acumulada
    ax.plot(
        range(1, len(explained_variance_ratio) + 1),
        np.cumsum(explained_variance_ratio),
        marker='o',
        color=COLORS['secondary'],
        label='Varianza acumulada'
    )
    
    # Línea de referencia en 80% de varianza explicada
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Umbral 80%')
    
    # Etiquetas y título
    ax.set_xlabel('Número de componente')
    ax.set_ylabel('Proporción de varianza explicada')
    ax.set_title('Gráfico de sedimentación (Scree Plot)')
    ax.set_xticks(range(1, len(explained_variance_ratio) + 1))
    ax.set_ylim(0, 1.05)
    
    # Añadir porcentajes en las barras
    for i, v in enumerate(explained_variance_ratio):
        ax.text(i + 1, v + 0.01, f'{v:.1%}', ha='center')
    
    ax.legend()
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def plot_biplot(X_pca, loadings, feature_names, output_file=None, n_features=None):
    """
    Genera un biplot para visualizar las relaciones entre variables y observaciones.
    
    Args:
        X_pca (array): Datos transformados por PCA.
        loadings (array): Cargas factoriales.
        feature_names (list): Nombres de las variables originales.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        n_features (int, optional): Número máximo de variables a mostrar. Si es None, muestra todas.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Seleccionar solo los primeros dos componentes
    X_pca_2d = X_pca[:, :2]
    loadings_2d = loadings[:, :2]
    
    # Limitar número de variables si es necesario
    if n_features is not None and n_features < len(feature_names):
        # Seleccionar las variables con mayor carga (suma de cuadrados)
        importance = np.sum(loadings_2d ** 2, axis=1)
        top_indices = np.argsort(importance)[-n_features:]
        loadings_2d = loadings_2d[top_indices]
        feature_names = [feature_names[i] for i in top_indices]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Escalar los datos para visualización
    scale = 1.0 / (X_pca_2d.max(axis=0) - X_pca_2d.min(axis=0))
    X_pca_scaled = X_pca_2d * scale
    
    # Graficar observaciones
    ax.scatter(
        X_pca_scaled[:, 0],
        X_pca_scaled[:, 1],
        alpha=0.3,
        color=COLORS['primary'],
        label='Observaciones'
    )
    
    # Graficar vectores de carga
    for i, (name, loading) in enumerate(zip(feature_names, loadings_2d)):
        ax.arrow(
            0, 0,
            loading[0], loading[1],
            head_width=0.05,
            head_length=0.05,
            fc=COLORS['secondary'],
            ec=COLORS['secondary']
        )
        ax.text(
            loading[0] * 1.15,
            loading[1] * 1.15,
            name,
            color=COLORS['secondary'],
            ha='center',
            va='center',
            fontsize=8
        )
    
    # Etiquetas y título
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_title('Biplot: Proyección de variables y observaciones')
    
    # Añadir círculo de correlación
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='gray', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    # Ajustar límites para mantener aspecto cuadrado
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def plot_loadings_heatmap(loadings, feature_names, component_names=None, output_file=None):
    """
    Genera un mapa de calor para visualizar las cargas factoriales.
    
    Args:
        loadings (array): Cargas factoriales.
        feature_names (list): Nombres de las variables originales.
        component_names (list, optional): Nombres de los componentes. Si es None, se generan automáticamente.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Crear nombres de componentes si no se proporcionan
    if component_names is None:
        component_names = [f'CP{i+1}' for i in range(loadings.shape[1])]
    
    # Crear DataFrame para facilitar la visualización
    loadings_df = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=component_names
    )
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Generar mapa de calor
    sns.heatmap(
        loadings_df,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        linewidths=0.5,
        ax=ax
    )
    
    # Etiquetas y título
    ax.set_title('Mapa de calor de cargas factoriales')
    ax.set_xlabel('Componentes Principales')
    ax.set_ylabel('Variables')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def pca_academic_performance(df):
    """
    Realiza análisis de componentes principales sobre las variables de rendimiento académico.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        
    Returns:
        tuple: (pca_model, X_pca, explained_variance_ratio, loadings, feature_names)
    """
    # Seleccionar variables de rendimiento académico numéricas
    academic_vars = [var for var in ACADEMIC_VARS if 'PUNT' in var]
    
    # Realizar PCA
    pca, X_pca, explained_variance, loadings = perform_pca(df, academic_vars)
    
    # Generar y guardar gráficos
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Scree plot
    scree_file = FIGURES_DIR / 'pca_academic_scree.png'
    plot_scree(explained_variance, output_file=scree_file)
    
    # Biplot
    biplot_file = FIGURES_DIR / 'pca_academic_biplot.png'
    plot_biplot(X_pca, loadings, academic_vars, output_file=biplot_file)
    
    # Heatmap de cargas
    heatmap_file = FIGURES_DIR / 'pca_academic_loadings.png'
    plot_loadings_heatmap(loadings, academic_vars, output_file=heatmap_file)
    
    # Guardar componentes en el DataFrame original
    df_pca = df.copy()
    for i in range(X_pca.shape[1]):
        df_pca[f'PCA_ACADEMIC_{i+1}'] = X_pca[:, i]
    
    # Guardar resultados
    pca_results_file = PROCESSED_DATA_DIR / 'pca_academic_results.csv'
    df_pca.to_csv(pca_results_file, index=False)
    print(f"Resultados de PCA guardados en {pca_results_file}")
    
    return pca, X_pca, explained_variance, loadings, academic_vars

if __name__ == "__main__":
    # Importar módulo de carga de datos
    from src.data.data_loader import get_data
    import os
    
    # Cargar datos
    df = get_data()
    
    # Realizar PCA sobre rendimiento académico
    if df is not None:
        pca_academic_performance(df)
