"""
Módulo para análisis de clustering jerárquico de los datos de Saber Pro.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import sys
from pathlib import Path
import os
import logging
import joblib # Import joblib

# Configurar logger
logger = logging.getLogger(__name__)

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.constants import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR, # Import MODELS_DIR
    FIGURE_SIZE,
    FIGURE_DPI,
    FIGURE_FORMAT,
    N_CLUSTERS_RANGE,
    RANDOM_STATE,
    COLORS,
    STRATA_COLORS
)

def perform_hierarchical_clustering(df, variables, n_clusters=None, scale=True, max_samples=5000):
    """
    Realiza clustering jerárquico sobre las variables seleccionadas.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        variables (list): Lista de nombres de columnas para incluir en el análisis.
        n_clusters (int, optional): Número de clusters a formar. Si es None, se determina automáticamente.
        scale (bool, optional): Si es True, estandariza las variables antes del análisis.
        max_samples (int, optional): Número máximo de muestras a utilizar para el clustering para evitar
                                     problemas de memoria. Por defecto 5000.
        
    Returns:
        tuple: (model, labels, X)
            - model: Modelo de clustering entrenado
            - labels: Etiquetas de cluster asignadas a cada observación
            - X: Datos utilizados para el clustering
    """
    # Seleccionar solo las variables numéricas
    X = df[variables].select_dtypes(include=['number'])
    
    # Manejar valores faltantes
    X = X.fillna(X.mean())
    
    # Aplicar muestreo si el conjunto de datos es demasiado grande
    original_size = X.shape[0]
    if original_size > max_samples:
        print(f"Aplicando muestreo para clustering jerárquico: reduciendo de {original_size} a {max_samples} muestras")
        X = X.sample(n=max_samples, random_state=RANDOM_STATE)
    
    # Estandarizar si es necesario
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Determinar número de clusters si no se especifica
    if n_clusters is None:
        n_clusters = 5  # Valor predeterminado
    
    # Realizar clustering jerárquico con política de memoria eficiente
    try:
        # Intentar usar memory="whatever" para optimizar memoria si está disponible
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            memory="whatever"  # Permite que scikit-learn gestione la memoria eficientemente
        )
    except TypeError:
        # Fallback si la versión de scikit-learn no soporta este parámetro
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
    
    labels = model.fit_predict(X_scaled)
    
    return model, labels, X_scaled

def plot_dendrogram(X, max_samples=100, output_file=None):
    """
    Genera un dendrograma para visualizar la estructura jerárquica.
    
    Args:
        X (array): Datos para el clustering.
        max_samples (int, optional): Número máximo de muestras a incluir en el dendrograma.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Limitar número de muestras si es necesario
    if X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Calcular linkage
    Z = linkage(X_sample, method='ward')
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[0] * 0.7))
    
    # Generar dendrograma
    dendrogram(
        Z,
        ax=ax,
        leaf_rotation=90.,
        leaf_font_size=8.,
        color_threshold=0.7 * max(Z[:, 2])
    )
    
    # Etiquetas y título
    ax.set_title('Dendrograma de Clustering Jerárquico')
    ax.set_xlabel('Muestra')
    ax.set_ylabel('Distancia')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def plot_silhouette_scores(df, variables, max_clusters=10, output_file=None, sample_size=5000):
    """
    Genera un gráfico de puntuaciones de silueta para diferentes números de clusters.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        variables (list): Lista de nombres de columnas para incluir en el análisis.
        max_clusters (int, optional): Número máximo de clusters a evaluar.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        sample_size (int, optional): Tamaño máximo de la muestra para usar. Por defecto 5000 para evitar problemas de memoria.
        
    Returns:
        tuple: (matplotlib.figure.Figure, optimal_n_clusters)
            - fig: Objeto figura con el gráfico
            - optimal_n_clusters: Número óptimo de clusters según el coeficiente de silueta
    """
    from sklearn.metrics import silhouette_score
    
    # Tomar una muestra si el DataFrame es muy grande para evitar problemas de memoria
    if df.shape[0] > sample_size:
        print(f"Tomando una muestra aleatoria de {sample_size} filas para el análisis de clustering")
        df_sample = df.sample(sample_size, random_state=RANDOM_STATE)
    else:
        df_sample = df
    
    # Seleccionar solo las variables numéricas
    X = df_sample[variables].select_dtypes(include=['number'])
    
    # Manejar valores faltantes
    X = X.fillna(X.mean())
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calcular puntuaciones de silueta para diferentes números de clusters
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        # Usar configuración compatible con versiones más recientes de scikit-learn
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = model.fit_predict(X_scaled)
        
        # Calcular puntuación de silueta
        try:
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        except:
            silhouette_scores.append(0)
    
    # Encontrar número óptimo de clusters
    optimal_n_clusters = np.argmax(silhouette_scores) + 2  # +2 porque empezamos desde 2 clusters
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Graficar puntuaciones de silueta
    ax.plot(
        range(2, max_clusters + 1),
        silhouette_scores,
        marker='o',
        color=COLORS['primary']
    )
    
    # Marcar número óptimo de clusters
    ax.axvline(
        x=optimal_n_clusters,
        color=COLORS['secondary'],
        linestyle='--',
        alpha=0.7,
        label=f'Óptimo: {optimal_n_clusters} clusters'
    )
    
    # Etiquetas y título
    ax.set_xlabel('Número de clusters')
    ax.set_ylabel('Puntuación de silueta')
    ax.set_title('Puntuaciones de silueta para diferentes números de clusters')
    ax.set_xticks(range(2, max_clusters + 1))
    ax.legend()
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig, optimal_n_clusters

def plot_cluster_profiles(df, variables, labels, output_file=None):
    """
    Genera un gráfico de perfiles de cluster para visualizar las características de cada grupo.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        variables (list): Lista de nombres de columnas para incluir en el análisis.
        labels (array): Etiquetas de cluster asignadas a cada observación.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Seleccionar solo las variables numéricas
    X = df[variables].select_dtypes(include=['number'])
    
    # Manejar valores faltantes
    X = X.fillna(X.mean())
    
    # Estandarizar para comparabilidad
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    # Añadir etiquetas de cluster
    X_scaled['Cluster'] = labels
    
    # Calcular medias por cluster
    cluster_means = X_scaled.groupby('Cluster').mean()
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Graficar perfiles
    for i, (idx, row) in enumerate(cluster_means.iterrows()):
        ax.plot(
            range(len(row)),
            row,
            marker='o',
            linewidth=2,
            label=f'Cluster {idx}'
        )
    
    # Etiquetas y título
    ax.set_xlabel('Variable')
    ax.set_ylabel('Valor estandarizado')
    ax.set_title('Perfiles de clusters')
    ax.set_xticks(range(len(cluster_means.columns)))
    ax.set_xticklabels(cluster_means.columns, rotation=90)
    ax.legend()
    
    # Añadir línea de referencia en 0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def plot_cluster_distribution(df, labels, group_var, output_file=None):
    """
    Genera un gráfico de distribución de clusters por una variable de agrupación.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        labels (array): Etiquetas de cluster asignadas a cada observación.
        group_var (str): Variable para agrupar (ej. estrato).
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Crear DataFrame con etiquetas
    df_clusters = df.copy()
    df_clusters['Cluster'] = labels
    
    # Verificar si la variable de agrupación existe
    if group_var not in df_clusters.columns:
        print(f"Variable {group_var} no encontrada en el DataFrame")
        return None
    
    # Calcular distribución
    cross_tab = pd.crosstab(
        df_clusters[group_var],
        df_clusters['Cluster'],
        normalize='index'
    ) * 100  # Convertir a porcentaje
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Graficar distribución
    cross_tab.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        colormap='tab10'
    )
    
    # Etiquetas y título
    ax.set_xlabel(group_var)
    ax.set_ylabel('Porcentaje')
    ax.set_title(f'Distribución de clusters por {group_var}')
    ax.legend(title='Cluster')
    
    # Añadir porcentajes
    for c in ax.containers:
        labels = [f'{v:.1f}%' if v > 5 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def cluster_socioeconomic_academic(df, sample_size=2000, max_clusters=6):
    """
    Realiza clustering jerárquico combinando variables socioeconómicas y académicas.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        sample_size (int, optional): Tamaño máximo de la muestra para usar. Por defecto 2000 para evitar problemas de memoria.
        max_clusters (int, optional): Número máximo de clusters a evaluar. Por defecto 6.
        
    Returns:
        tuple: (model, labels, X_scaled)
    """
    # Para el análisis silhouette y otros procedimientos preparatorios, usamos una muestra
    # pero mantenemos el DataFrame completo para aplicar el modelo final
    df_full = df.copy()
    
    # Tomar una muestra para los análisis previos (silhouette, etc.)
    if df.shape[0] > sample_size:
        print(f"Tomando una muestra aleatoria de {sample_size} filas para el análisis previo de clustering")
        df_sample = df.sample(sample_size, random_state=RANDOM_STATE)
    else:
        df_sample = df.copy()
        
    print(f"Dimensiones del DataFrame muestreado para clustering: {df_sample.shape}")
    
    # Seleccionar variables para clustering
    socioeconomic_vars = [
        'NSE_SCORE',  # Variable compuesta de nivel socioeconómico
        'FAMI_EDUCACIONPADRE_NIVEL',
        'FAMI_EDUCACIONMADRE_NIVEL',
        'ESTRATO_NUM'
    ]
    
    academic_vars = [var for var in df_sample.columns if var.startswith('MOD_') and 'PUNT' in var]
    
    # Filtrar solo las variables que existen en el DataFrame
    cluster_vars = [var for var in socioeconomic_vars + academic_vars if var in df_sample.columns]
    print(f"Variables usadas para clustering: {cluster_vars}")
    
    # Determinar número óptimo de clusters
    os.makedirs(FIGURES_DIR, exist_ok=True)
    silhouette_file = FIGURES_DIR / 'cluster_silhouette_scores.png'
    _, optimal_n_clusters = plot_silhouette_scores(df_sample, cluster_vars, max_clusters=max_clusters, 
                                                output_file=silhouette_file, sample_size=min(sample_size, 5000))
    
    print(f"Número óptimo de clusters determinado: {optimal_n_clusters}")
    
    # Realizar clustering con número óptimo de clusters
    # Usamos el parámetro max_samples para evitar problemas de memoria durante el clustering jerárquico
    model, labels_sample, X_scaled = perform_hierarchical_clustering(
        df_sample, 
        cluster_vars, 
        n_clusters=optimal_n_clusters,
        max_samples=min(sample_size, 5000)
    )
    
    # Generar dendrograma
    dendrogram_file = FIGURES_DIR / 'cluster_dendrogram.png'
    plot_dendrogram(X_scaled, max_samples=min(100, sample_size), output_file=dendrogram_file)
    
    # Generar perfiles de clusters
    profiles_file = FIGURES_DIR / 'cluster_profiles.png'
    plot_cluster_profiles(df_sample, cluster_vars, labels_sample, output_file=profiles_file)
    
    # Generar distribución por estrato
    if 'FAMI_ESTRATOVIVIENDA' in df_sample.columns:
        strata_dist_file = FIGURES_DIR / 'cluster_distribution_by_strata.png'
        plot_cluster_distribution(df_sample, labels_sample, 'FAMI_ESTRATOVIVIENDA', output_file=strata_dist_file)
    
    # Aplicar el modelo a todos los datos para obtener etiquetas para el conjunto completo
    # Primero entrenar un clasificador rápido con las etiquetas generadas
    from sklearn.ensemble import RandomForestClassifier
    
    # Seleccionar solo las variables numéricas existentes en todo el conjunto de datos
    X_train = df_sample[cluster_vars].select_dtypes(include=['number']).fillna(0)
    
    # Entrenar o cargar el clasificador para propagar etiquetas
    classifier_model_path = MODELS_DIR / "cluster_label_classifier.joblib"
    clf = None
    if os.path.exists(classifier_model_path):
        try:
            logger.info(f"Cargando clasificador de etiquetas de cluster existente desde {classifier_model_path}")
            clf = joblib.load(classifier_model_path)
        except Exception as e:
            logger.error(f"Error al cargar el clasificador de etiquetas de cluster desde {classifier_model_path}: {e}. Se entrenará uno nuevo.", exc_info=True)
            clf = None # Asegurar que se entrene uno nuevo

    if clf is None:
        logger.info("Entrenando nuevo clasificador RandomForest para propagación de etiquetas de cluster...")
        clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        try:
            clf.fit(X_train, labels_sample)
            logger.info("Clasificador RandomForest entrenado.")
            try:
                os.makedirs(MODELS_DIR, exist_ok=True) # Asegurar que el directorio de modelos exista
                joblib.dump(clf, classifier_model_path)
                logger.info(f"Clasificador RandomForest guardado en {classifier_model_path}")
            except Exception as e:
                logger.error(f"Error al guardar el clasificador RandomForest en {classifier_model_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error al entrenar el clasificador RandomForest para propagación de etiquetas: {e}", exc_info=True)
            # No se puede continuar si el clasificador no se entrena
            return model, labels_sample, X_scaled # Devuelve resultados del clustering en la muestra
            
    if clf is None: # Doble chequeo por si el entrenamiento falló y no se retornó antes
        logger.error("El clasificador RandomForest no está disponible. No se pueden propagar las etiquetas al conjunto completo.")
        return model, labels_sample, X_scaled

    # Aplicar el clasificador a todo el conjunto de datos en lotes para evitar problemas de memoria
    logger.info(f"Aplicando modelo clasificador a todos los datos ({df_full.shape[0]} filas) en lotes")
    batch_size = 10000  # Procesar en lotes de 10,000 filas
    all_labels = []
    
    for i in range(0, df_full.shape[0], batch_size):
        end_idx = min(i + batch_size, df_full.shape[0])
        print(f"Procesando lote {i//batch_size + 1}: filas {i} a {end_idx}")
        
        # Tomar un lote de datos
        df_batch = df_full.iloc[i:end_idx]
        
        # Preparar datos para predicción
        X_batch = df_batch[cluster_vars].select_dtypes(include=['number']).fillna(0)
        
        # Predecir cluster
        batch_labels = clf.predict(X_batch)
        all_labels.extend(batch_labels)
    
    # Convertir todas las etiquetas a una array de NumPy
    labels = np.array(all_labels)
    
    print(f"Etiquetas generadas para {len(labels)} instancias en el conjunto completo")
    
    # Guardar etiquetas en el DataFrame original
    df_clusters = df_full.copy()
    df_clusters['CLUSTER'] = labels
    
    # Guardar resultados
    clusters_file = PROCESSED_DATA_DIR / 'clustering_results.csv'
    df_clusters.to_csv(clusters_file, index=False)
    print(f"Resultados de clustering guardados en {clusters_file}")
    
    return model, labels, X_scaled

if __name__ == "__main__":
    # Importar módulo de carga de datos
    from src.data.data_loader import get_data # Movido aquí para que el logger de este módulo esté configurado
    
    # Configuración básica de logging para pruebas
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("clustering_analysis_test.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.info("Ejecutando clustering_analysis.py como script principal para prueba.")
    
    # Cargar datos
    logger.info("Cargando datos para la prueba de clustering...")
    df = get_data()
    
    # Realizar clustering
    if df is not None:
        logger.info("Datos cargados, procediendo con cluster_socioeconomic_academic.")
        cluster_socioeconomic_academic(df)
    else:
        logger.error("No se pudieron cargar los datos para la prueba de clustering.")
