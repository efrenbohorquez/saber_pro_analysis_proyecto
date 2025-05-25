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
    logger.info("Iniciando preparación de datos categóricos para MCA.")
    for col in X.columns:
        logger.debug(f"Procesando columna para MCA: {col}")
        if X[col].dtype.kind in 'ifc':  # integer, float, complex
            try:
                num_unique = X[col].nunique(dropna=True)
                logger.debug(f"Columna '{col}' es numérica con {num_unique} valores únicos.")
                if num_unique <= 5: # Si pocos valores únicos, tratar como categórica directamente
                    X[col] = X[col].astype('category')
                    logger.info(f"Columna '{col}' convertida a 'category' directamente.")
                else:
                    n_bins = min(5, max(2, num_unique - 1)) # Asegurar al menos 2 bins
                    X[col] = pd.qcut(X[col], n_bins, labels=[f'{col}_Q{i+1}' for i in range(n_bins)], duplicates='drop')
                    logger.info(f"Columna '{col}' discretizada en {n_bins} bins usando qcut. Nuevas categorías: {X[col].cat.categories.tolist()}")
            except ValueError as ve:
                logger.warning(f"Error de ValueError al discretizar columna '{col}': {ve}. Se convertirá a string.", exc_info=True)
                X[col] = X[col].astype('str') # Fallback si qcut falla
            except Exception as e: # Captura más general por si acaso
                logger.error(f"Error inesperado al procesar columna numérica '{col}': {e}. Se convertirá a string.", exc_info=True)
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
    logger.info(f"Iniciando MCA con n_components={n_components} para variables: {variables}")
    X = prepare_categorical_data(df, variables)
    if X is None or X.empty:
        logger.error("La preparación de datos categóricos no devolvió datos válidos. Abortando MCA.")
        return None, None, None
    logger.info(f"Forma de los datos preparados para MCA (X): {X.shape}")
    
    # Realizar MCA
    try:
        mca = prince.MCA(
            n_components=n_components,
            n_iter=10,
            copy=True,
            check_input=True,
            engine='sklearn',
            random_state=RANDOM_STATE
        )
        
        logger.debug("Ajustando modelo MCA...")
        mca = mca.fit(X)
        logger.info("Modelo MCA ajustado.")
        
        logger.debug("Transformando datos con modelo MCA...")
        X_mca = mca.transform(X)
        logger.info(f"Datos transformados con MCA. Forma de X_mca: {X_mca.shape}")
        
        # Calcular la inercia explicada
        explained_inertia = []
        if hasattr(mca, 'eigenvalues_') and mca.eigenvalues_ is not None:
            total_inertia = sum(mca.eigenvalues_)
            if total_inertia > 0:
                explained_inertia = [val/total_inertia for val in mca.eigenvalues_]
                logger.info(f"Inercia explicada por cada componente: {explained_inertia}")
                logger.info(f"Inercia explicada acumulada: {np.cumsum(explained_inertia).tolist()}")
            else:
                logger.warning("Inercia total es cero, no se puede calcular la proporción de inercia explicada.")
                explained_inertia = [0.0] * n_components # O alguna otra indicación de error/valor por defecto
        else:
            logger.warning("Atributo 'eigenvalues_' no encontrado o es None en el objeto MCA. Usando inercia aproximada.")
            explained_inertia = [1.0/n_components if n_components > 0 else 0.0] * n_components
        
        return mca, X_mca, explained_inertia
        
    except (ValueError, TypeError) as e: # Errores comunes de scikit-learn/prince con datos inadecuados
        logger.error(f"Error de ValueError/TypeError durante MCA fit/transform: {e}", exc_info=True)
        return None, None, None
    except Exception as e: # Captura general para otros errores de la librería Prince
        logger.error(f"Error inesperado durante la ejecución de MCA: {e}", exc_info=True)
        return None, None, None

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
        try:
            plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"MCA scree plot guardado en: {output_file}")
        except (IOError, OSError) as e:
            logger.error(f"Error al guardar MCA scree plot en {output_file}: {e}", exc_info=True)
    
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
    logger.info("Generando mapa factorial MCA.")
    coords = None
    try:
        if not hasattr(mca, 'column_coordinates_'):
            logger.warning("El objeto MCA no tiene el atributo 'column_coordinates_'. No se puede generar el mapa factorial.")
            # Intentar acceder a eigenvalues para verificar si el modelo está ajustado, si no, es un problema mayor.
            if not hasattr(mca, 'eigenvalues_'):
                 logger.error("El objeto MCA parece no estar ajustado correctamente (sin eigenvalues_).")
            return None # Devuelve None si no se puede generar el gráfico

        coords = mca.column_coordinates_
        logger.debug(f"Coordenadas de columnas obtenidas. Forma: {coords.shape if coords is not None else 'None'}")

        if coords is None: # Doble chequeo por si acaso
             logger.error("mca.column_coordinates_ devolvió None. No se puede generar el mapa factorial.")
             return None

        # Limitar número de categorías si es necesario
        if n_categories is not None and n_categories < len(coords):
            logger.info(f"Limitando el número de categorías en el mapa factorial a {n_categories}.")
            try:
                coords = coords.sample(n=n_categories, random_state=RANDOM_STATE)
            except ValueError as ve: # Si n_categories es mayor que la población
                logger.warning(f"No se pudo muestrear {n_categories} categorías (total: {len(coords)}): {ve}. Usando todas las categorías.", exc_info=True)
        
    except AttributeError as ae:
        logger.error(f"Error de atributo al acceder a coordenadas/eigenvalues en el objeto MCA: {ae}", exc_info=True)
        return None
    except Exception as e: # Captura general para otros errores inesperados
        logger.error(f"Error inesperado al obtener/procesar coordenadas para el mapa factorial: {e}", exc_info=True)
        return None # Devuelve None si hay un error crítico
    
    if coords is None or coords.empty:
        logger.warning("No hay coordenadas de categorías para graficar en el mapa factorial. Skipping.")
        return None

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
        try:
            plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"MCA factor map guardado en: {output_file}")
        except (IOError, OSError) as e:
            logger.error(f"Error al guardar MCA factor map en {output_file}: {e}", exc_info=True)
        try:
            plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"MCA individuals plot guardado en: {output_file}")
        except (IOError, OSError) as e:
            logger.error(f"Error al guardar MCA individuals plot en {output_file}: {e}", exc_info=True)
    
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
    
    logger.info("Iniciando MCA sobre variables socioeconómicas.")
    logger.debug(f"Variables socioeconómicas para MCA: {socioeconomic_vars}")
    
    # Realizar MCA
    mca_results = perform_mca(df, socioeconomic_vars)
    if mca_results is None or mca_results[0] is None:
        logger.error("Falló la ejecución de perform_mca. No se pueden generar gráficos ni guardar resultados.")
        return None, None, None
    mca, X_mca, explained_inertia = mca_results
    logger.info("MCA principal completado.")

    # Generar y guardar gráficos
    try:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        logger.info(f"Directorio de figuras asegurado/creado: {FIGURES_DIR}")

        if explained_inertia is not None and len(explained_inertia) > 0:
            scree_file = FIGURES_DIR / 'mca_socioeconomic_scree.png'
            logger.info(f"Generando scree plot para MCA socioeconómico, guardando en {scree_file}")
            plot_mca_scree(explained_inertia, output_file=scree_file)
        else:
            logger.warning("No se generará el scree plot de MCA debido a que explained_inertia no está disponible.")

        if mca is not None and explained_inertia is not None and len(explained_inertia) >= 2:
            factor_map_file = FIGURES_DIR / 'mca_socioeconomic_factor_map.png'
            logger.info(f"Generando mapa factorial para MCA socioeconómico, guardando en {factor_map_file}")
            plot_mca_factor_map(mca, explained_inertia, output_file=factor_map_file)
        else:
            logger.warning("No se generará el mapa factorial de MCA debido a que el modelo MCA o explained_inertia (con al menos 2 componentes) no están disponibles.")

        if X_mca is not None and 'FAMI_ESTRATOVIVIENDA' in df.columns:
            individuals_file = FIGURES_DIR / 'mca_socioeconomic_individuals_by_strata.png'
            logger.info(f"Generando gráfico de individuos por estrato para MCA socioeconómico, guardando en {individuals_file}")
            plot_mca_individuals(X_mca, df, color_var='FAMI_ESTRATOVIVIENDA', output_file=individuals_file)
        elif X_mca is None:
             logger.warning("No se generará el gráfico de individuos por estrato porque X_mca no está disponible.")
        else:
            logger.warning("Columna 'FAMI_ESTRATOVIVIENDA' no encontrada, no se generará el gráfico de individuos por estrato.")
            
        if X_mca is not None and 'RA_NIVEL' in df.columns:
            individuals_ra_file = FIGURES_DIR / 'mca_socioeconomic_individuals_by_ra.png'
            logger.info(f"Generando gráfico de individuos por RA_NIVEL para MCA socioeconómico, guardando en {individuals_ra_file}")
            plot_mca_individuals(X_mca, df, color_var='RA_NIVEL', output_file=individuals_ra_file)
        elif X_mca is None:
             logger.warning("No se generará el gráfico de individuos por RA_NIVEL porque X_mca no está disponible.")
        else:
            logger.warning("Columna 'RA_NIVEL' no encontrada, no se generará el gráfico de individuos por RA_NIVEL.")
            
    except Exception as e:
        logger.error(f"Error durante la generación o guardado de gráficos MCA: {e}", exc_info=True)

    # Guardar coordenadas en el DataFrame original
    if X_mca is not None:
        try:
            df_mca = df.copy()
            for i in range(X_mca.shape[1]):
                df_mca[f'MCA_SOCIOECONOMIC_{i+1}'] = X_mca.iloc[:, i]
            
            mca_results_file = PROCESSED_DATA_DIR / 'mca_socioeconomic_results.csv'
            df_mca.to_csv(mca_results_file, index=False)
            logger.info(f"Resultados de MCA (coordenadas) guardados en {mca_results_file}")
        except (IOError, OSError) as e:
            logger.error(f"Error de E/S al guardar los resultados de MCA en {mca_results_file}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error inesperado al guardar los resultados de MCA: {e}", exc_info=True)
    else:
        logger.warning("X_mca es None, no se guardarán los resultados de MCA.")
        
    logger.info("Análisis MCA sobre variables socioeconómicas finalizado.")
    return mca, X_mca, explained_inertia

if __name__ == "__main__":
    # Configuración básica de logging para pruebas
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("mca_analysis_test.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.info("Ejecutando mca_analysis.py como script principal para prueba.")
    # Importar módulo de carga de datos
    from src.data.data_loader import get_data # Movido aquí para que el logger de este módulo esté configurado
    
    # Cargar datos
    logger.info("Cargando datos para la prueba de MCA...")
    df = get_data()
    
    # Realizar MCA sobre variables socioeconómicas
    if df is not None:
        logger.info("Datos cargados, procediendo con mca_socioeconomic.")
        mca_socioeconomic(df)
    else:
        logger.error("No se pudieron cargar los datos para la prueba de MCA.")
