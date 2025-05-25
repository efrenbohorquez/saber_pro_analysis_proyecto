"""
Módulo para modelos predictivos sobre los datos de Saber Pro.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
from pathlib import Path
import os
import joblib # Import joblib
import logging

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
    RANDOM_STATE,
    TEST_SIZE,
    COLORS
)

def prepare_predictive_data(df, target_var, predictor_vars=None):
    """
    Prepara los datos para modelos predictivos.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        target_var (str): Variable objetivo a predecir.
        predictor_vars (list, optional): Lista de variables predictoras. Si es None, se usan todas las numéricas.
        
    Returns:
        tuple: (X, y, X_train, X_test, y_train, y_test)
    """
    # Verificar si la variable objetivo existe
    if target_var not in df.columns:
        print(f"Variable objetivo {target_var} no encontrada en el DataFrame")
        return None, None, None, None, None, None
    
    # Seleccionar variables predictoras
    if predictor_vars is None:
        # Usar todas las variables numéricas excepto la objetivo
        X = df.select_dtypes(include=['number']).drop(columns=[target_var], errors='ignore')
    else:
        # Filtrar solo las variables que existen en el DataFrame
        valid_predictors = [var for var in predictor_vars if var in df.columns]
        X = df[valid_predictors]
    
    # Seleccionar variable objetivo
    y = df[target_var]
    
    # Manejar valores faltantes
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    return X, y, X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo de regresión lineal.
    
    Args:
        X_train (pandas.DataFrame): Variables predictoras de entrenamiento.
        y_train (pandas.Series): Variable objetivo de entrenamiento.
        X_test (pandas.DataFrame): Variables predictoras de prueba.
        y_test (pandas.Series): Variable objetivo de prueba.
        
    Returns:
        tuple: (model, y_pred, metrics)
            - model: Modelo entrenado
            - y_pred: Predicciones en conjunto de prueba
            - metrics: Diccionario con métricas de evaluación
    """
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    return model, y_pred, metrics

def train_ridge_regression(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo de regresión Ridge con selección de hiperparámetros.
    
    Args:
        X_train (pandas.DataFrame): Variables predictoras de entrenamiento.
        y_train (pandas.Series): Variable objetivo de entrenamiento.
        X_test (pandas.DataFrame): Variables predictoras de prueba.
        y_test (pandas.Series): Variable objetivo de prueba.
        
    Returns:
        tuple: (model, y_pred, metrics)
    """
    # Definir grid de hiperparámetros
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    logger.debug(f"Grid de parámetros para Ridge: {param_grid}, CV=5")
    
    # Buscar mejores hiperparámetros
    grid_search = GridSearchCV(
        Ridge(random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1 # Usar todos los cores
    )
    
    grid_search.fit(X_train, y_train)
    logger.info(f"Mejores parámetros para Ridge: {grid_search.best_params_}")
    
    # Obtener mejor modelo
    model = grid_search.best_estimator_
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'best_alpha': grid_search.best_params_['alpha']
    }
    
    return model, y_pred, metrics

def train_lasso_regression(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo de regresión Lasso con selección de hiperparámetros.
    
    Args:
        X_train (pandas.DataFrame): Variables predictoras de entrenamiento.
        y_train (pandas.Series): Variable objetivo de entrenamiento.
        X_test (pandas.DataFrame): Variables predictoras de prueba.
        y_test (pandas.Series): Variable objetivo de prueba.
        
    Returns:
        tuple: (model, y_pred, metrics)
    """
    # Definir grid de hiperparámetros
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    logger.debug(f"Grid de parámetros para Lasso: {param_grid}, CV=5")
        
    # Buscar mejores hiperparámetros
    grid_search = GridSearchCV(
        Lasso(random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1 # Usar todos los cores
    )
    
    grid_search.fit(X_train, y_train)
    logger.info(f"Mejores parámetros para Lasso: {grid_search.best_params_}")
    
    # Obtener mejor modelo
    model = grid_search.best_estimator_
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'best_alpha': grid_search.best_params_['alpha']
    }
    
    return model, y_pred, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo de Random Forest con selección de hiperparámetros.
    
    Args:
        X_train (pandas.DataFrame): Variables predictoras de entrenamiento.
        y_train (pandas.Series): Variable objetivo de entrenamiento.
        X_test (pandas.DataFrame): Variables predictoras de prueba.
        y_test (pandas.Series): Variable objetivo de prueba.
        
    Returns:
        tuple: (model, y_pred, metrics)
    """
    # Definir grid de hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    logger.debug(f"Grid de parámetros para RandomForest: {param_grid}, CV=3 (en muestra si aplica)")
    
    # Buscar mejores hiperparámetros (con una muestra pequeña para eficiencia)
    if X_train.shape[0] > 1000:
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=1000, random_state=RANDOM_STATE
        )
    else:
        X_sample, y_sample = X_train, y_train
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid,
        cv=3,  # Reducir CV para eficiencia
        scoring='neg_mean_squared_error',
        n_jobs=-1  # Usar todos los núcleos
    )
    
    grid_search.fit(X_sample, y_sample)
    
    # Entrenar modelo final con mejores hiperparámetros
    model = RandomForestRegressor(
        **grid_search.best_params_,
        random_state=RANDOM_STATE,
        n_jobs=-1 # Usar todos los cores para el modelo final también
    )
    logger.info(f"Entrenando RandomForest final con parámetros: {grid_search.best_params_}")
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    
    return model, y_pred, metrics

def plot_feature_importance(model, feature_names, output_file=None, top_n=20):
    """
    Genera un gráfico de importancia de variables para modelos que lo soportan.
    
    Args:
        model: Modelo entrenado con atributo feature_importances_ o coef_.
        feature_names (list): Nombres de las variables.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        top_n (int, optional): Número máximo de variables a mostrar.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Determinar tipo de modelo y obtener importancias
    if hasattr(model, 'feature_importances_'):
        # Random Forest y otros modelos basados en árboles
        importances = model.feature_importances_
        title = 'Importancia de variables (Random Forest)'
    elif hasattr(model, 'coef_'):
        # Modelos lineales
        importances = np.abs(model.coef_)
        title = 'Coeficientes absolutos (Modelo Lineal)'
    else:
        print("El modelo no tiene atributo feature_importances_ o coef_")
        return None
    
    # Crear DataFrame para ordenar
    importance_df = pd.DataFrame({
        'Variable': feature_names,
        'Importancia': importances
    })
    
    # Ordenar por importancia
    importance_df = importance_df.sort_values('Importancia', ascending=False)
    
    # Limitar número de variables
    if top_n is not None and top_n < len(importance_df):
        importance_df = importance_df.head(top_n)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Graficar importancias
    sns.barplot(
        x='Importancia',
        y='Variable',
        data=importance_df,
        ax=ax,
        color=COLORS['primary']
    )
    
    # Etiquetas y título
    ax.set_title(title)
    ax.set_xlabel('Importancia')
    ax.set_ylabel('Variable')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def plot_prediction_vs_actual(y_test, y_pred, output_file=None):
    """
    Genera un gráfico de dispersión de valores predichos vs. reales.
    
    Args:
        y_test (array): Valores reales.
        y_pred (array): Valores predichos.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Graficar dispersión
    ax.scatter(
        y_test,
        y_pred,
        alpha=0.5,
        color=COLORS['primary']
    )
    
    # Añadir línea de referencia (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--',
        lw=2
    )
    
    # Etiquetas y título
    ax.set_xlabel('Valor real')
    ax.set_ylabel('Valor predicho')
    ax.set_title('Predicción vs. Valor real')
    
    # Añadir métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ax.text(
        0.05, 0.95,
        f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def plot_residuals(y_test, y_pred, output_file=None):
    """
    Genera un gráfico de residuos.
    
    Args:
        y_test (array): Valores reales.
        y_pred (array): Valores predichos.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Calcular residuos
    residuals = y_test - y_pred
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # Gráfico de dispersión de residuos
    ax1.scatter(
        y_pred,
        residuals,
        alpha=0.5,
        color=COLORS['primary']
    )
    
    # Añadir línea de referencia en y = 0
    ax1.axhline(y=0, color='r', linestyle='--')
    
    # Etiquetas y título
    ax1.set_xlabel('Valor predicho')
    ax1.set_ylabel('Residuo')
    ax1.set_title('Residuos vs. Valores predichos')
    
    # Histograma de residuos
    sns.histplot(
        residuals,
        kde=True,
        ax=ax2,
        color=COLORS['primary']
    )
    
    # Etiquetas y título
    ax2.set_xlabel('Residuo')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de residuos')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def compare_models(models_dict, X_test, y_test, output_file=None):
    """
    Compara varios modelos y genera un gráfico de barras con métricas.
    
    Args:
        models_dict (dict): Diccionario con nombres de modelos como claves y tuplas (modelo, métricas) como valores.
        X_test (pandas.DataFrame): Variables predictoras de prueba.
        y_test (pandas.Series): Variable objetivo de prueba.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        tuple: (matplotlib.figure.Figure, pandas.DataFrame)
            - fig: Objeto figura con el gráfico
            - comparison_df: DataFrame con comparación de métricas
    """
    # Crear DataFrame para comparación
    comparison_data = []
    
    for name, (model, _, metrics) in models_dict.items():
        # Predecir con cada modelo
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Añadir a datos de comparación
        comparison_data.append({
            'Modelo': name,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae
        })
    
    # Crear DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Crear figura con tres subplots
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1]))
    
    # Graficar R²
    sns.barplot(
        x='Modelo',
        y='R²',
        data=comparison_df,
        ax=axes[0],
        palette='viridis'
    )
    axes[0].set_title('R² (mayor es mejor)')
    axes[0].set_ylim(0, 1)
    
    # Graficar RMSE
    sns.barplot(
        x='Modelo',
        y='RMSE',
        data=comparison_df,
        ax=axes[1],
        palette='viridis'
    )
    axes[1].set_title('RMSE (menor es mejor)')
    
    # Graficar MAE
    sns.barplot(
        x='Modelo',
        y='MAE',
        data=comparison_df,
        ax=axes[2],
        palette='viridis'
    )
    axes[2].set_title('MAE (menor es mejor)')
    
    # Rotar etiquetas
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig, comparison_df

def predict_academic_performance(df):
    """
    Entrena modelos predictivos para rendimiento académico basados en variables socioeconómicas.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        
    Returns:
        dict: Diccionario con modelos entrenados y métricas.
    """
    # Definir variable objetivo
    logger.info("Iniciando predicción de rendimiento académico.")
    target_var = 'MOD_RAZONA_CUANTITAT_PUNT'  # Puntaje en razonamiento cuantitativo
    logger.info(f"Variable objetivo: {target_var}")
    
    # Definir variables predictoras (socioeconómicas)
    predictor_vars = [
        'NSE_SCORE',
        'FAMI_EDUCACIONPADRE_NIVEL',
        'FAMI_EDUCACIONMADRE_NIVEL',
        'ESTRATO_NUM',
        'FAMI_TIENECOMPUTADOR',
        'FAMI_TIENEINTERNET',
        'FAMI_TIENELAVADORA',
        'FAMI_TIENEAUTOMOVIL',
        'ESTU_HORASSEMANATRABAJA_NUM'
    ]
    
    # Filtrar solo las variables que existen en el DataFrame
    predictor_vars = [var for var in predictor_vars if var in df.columns]
    
    # Preparar datos
    logger.info("Preparando datos para modelos predictivos...")
    prep_data_result = prepare_predictive_data(df, target_var, predictor_vars)
    
    if prep_data_result is None or prep_data_result[0] is None:
        logger.error("Falló la preparación de datos. Abortando predicción.")
        return None
    X, y, X_train, X_test, y_train, y_test = prep_data_result
    logger.info(f"Datos preparados. X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Crear directorios para figuras y modelos si no existen
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Directorios de figuras y modelos asegurados: {FIGURES_DIR}, {MODELS_DIR}")
    
    # Entrenar modelos
    models = {}
    model_predictions = {}

    # --- Regresión Lineal ---
    model_name = "Regresión Lineal"
    model_path = MODELS_DIR / "linear_regression_model.joblib"
    logger.info(f"--- {model_name} ---")
    if os.path.exists(model_path):
        logger.info(f"Cargando modelo existente desde {model_path}")
        lr_model = joblib.load(model_path)
        lr_pred = lr_model.predict(X_test)
        lr_metrics = {
            'r2': r2_score(y_test, lr_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
            'mae': mean_absolute_error(y_test, lr_pred)
        }
    else:
        logger.info(f"Entrenando nuevo modelo {model_name}...")
        lr_model, lr_pred, lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test)
        logger.info(f"Guardando modelo {model_name} en {model_path}")
        joblib.dump(lr_model, model_path)
    
    models[model_name] = (lr_model, lr_pred, lr_metrics) # Guardar métricas originales
    model_predictions[f'Pred_{model_name.replace(" ", "_")}'] = lr_pred
    logger.info(f"{model_name} - R2: {lr_metrics['r2']:.4f}, RMSE: {lr_metrics['rmse']:.4f}, MAE: {lr_metrics['mae']:.4f}")

    # Graficar importancia de variables
    lr_importance_file = FIGURES_DIR / 'lr_feature_importance.png'
    plot_feature_importance(lr_model, X.columns, output_file=lr_importance_file) # X.columns son los feature names
    
    # Graficar predicción vs. real
    lr_pred_file = FIGURES_DIR / 'lr_prediction_vs_actual.png'
    plot_prediction_vs_actual(y_test, lr_pred, output_file=lr_pred_file)
    
    # Graficar residuos
    lr_residuals_file = FIGURES_DIR / 'lr_residuals.png'
    plot_residuals(y_test, lr_pred, output_file=lr_residuals_file)

    # --- Ridge ---
    model_name = "Ridge"
    model_path = MODELS_DIR / "ridge_model.joblib"
    logger.info(f"--- {model_name} ---")
    if os.path.exists(model_path):
        logger.info(f"Cargando modelo existente desde {model_path}")
        ridge_model = joblib.load(model_path)
        ridge_pred = ridge_model.predict(X_test)
        ridge_metrics = {
            'r2': r2_score(y_test, ridge_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)),
            'mae': mean_absolute_error(y_test, ridge_pred),
            'best_alpha': ridge_model.alpha # Asumiendo que el alpha se guarda
        }
    else:
        logger.info(f"Entrenando nuevo modelo {model_name}...")
        ridge_model, ridge_pred, ridge_metrics = train_ridge_regression(X_train, y_train, X_test, y_test)
        logger.info(f"Guardando modelo {model_name} en {model_path}")
        joblib.dump(ridge_model, model_path)
        
    models[model_name] = (ridge_model, ridge_pred, ridge_metrics)
    model_predictions[f'Pred_{model_name.replace(" ", "_")}'] = ridge_pred
    logger.info(f"{model_name} - R2: {ridge_metrics['r2']:.4f}, RMSE: {ridge_metrics['rmse']:.4f}, MAE: {ridge_metrics['mae']:.4f}, Alpha: {ridge_metrics.get('best_alpha', 'N/A')}")

    # --- Lasso ---
    model_name = "Lasso"
    model_path = MODELS_DIR / "lasso_model.joblib"
    logger.info(f"--- {model_name} ---")
    if os.path.exists(model_path):
        logger.info(f"Cargando modelo existente desde {model_path}")
        lasso_model = joblib.load(model_path)
        lasso_pred = lasso_model.predict(X_test)
        lasso_metrics = {
            'r2': r2_score(y_test, lasso_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, lasso_pred)),
            'mae': mean_absolute_error(y_test, lasso_pred),
            'best_alpha': lasso_model.alpha # Asumiendo que el alpha se guarda
        }
    else:
        logger.info(f"Entrenando nuevo modelo {model_name}...")
        lasso_model, lasso_pred, lasso_metrics = train_lasso_regression(X_train, y_train, X_test, y_test)
        logger.info(f"Guardando modelo {model_name} en {model_path}")
        joblib.dump(lasso_model, model_path)

    models[model_name] = (lasso_model, lasso_pred, lasso_metrics)
    model_predictions[f'Pred_{model_name.replace(" ", "_")}'] = lasso_pred
    logger.info(f"{model_name} - R2: {lasso_metrics['r2']:.4f}, RMSE: {lasso_metrics['rmse']:.4f}, MAE: {lasso_metrics['mae']:.4f}, Alpha: {lasso_metrics.get('best_alpha', 'N/A')}")
    
    # --- Random Forest ---
    model_name = "Random Forest"
    model_path = MODELS_DIR / "random_forest_model.joblib"
    logger.info(f"--- {model_name} ---")
    if os.path.exists(model_path):
        logger.info(f"Cargando modelo existente desde {model_path}")
        rf_model = joblib.load(model_path)
        rf_pred = rf_model.predict(X_test)
        rf_metrics = {
            'r2': r2_score(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'mae': mean_absolute_error(y_test, rf_pred),
            'best_params': rf_model.get_params() # Guardar todos los params
        }
    else:
        logger.info(f"Entrenando nuevo modelo {model_name}...")
        rf_model, rf_pred, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
        logger.info(f"Guardando modelo {model_name} en {model_path}")
        joblib.dump(rf_model, model_path)

    models[model_name] = (rf_model, rf_pred, rf_metrics)
    model_predictions[f'Pred_{model_name.replace(" ", "_")}'] = rf_pred
    logger.info(f"{model_name} - R2: {rf_metrics['r2']:.4f}, RMSE: {rf_metrics['rmse']:.4f}, MAE: {rf_metrics['mae']:.4f}")
    logger.info(f"Mejores parámetros para {model_name}: {rf_metrics.get('best_params', 'N/A')}")
    
    # Graficar importancia de variables para Random Forest
    rf_importance_file = FIGURES_DIR / 'rf_feature_importance.png'
    plot_feature_importance(rf_model, X.columns, output_file=rf_importance_file)
    
    # Graficar predicción vs. real para Random Forest
    rf_pred_file = FIGURES_DIR / 'rf_prediction_vs_actual.png'
    plot_prediction_vs_actual(y_test, rf_pred, output_file=rf_pred_file)
    
    # Comparar modelos
    comparison_file = FIGURES_DIR / 'model_comparison.png'
    _, comparison_df = compare_models(models, X_test, y_test, output_file=comparison_file)
    
    # Guardar resultados de comparación
    comparison_csv = PROCESSED_DATA_DIR / 'model_comparison_metrics.csv'
    try:
        comparison_df.to_csv(comparison_csv, index=False)
        logger.info(f"Comparación de métricas de modelos guardada en {comparison_csv}")
    except (IOError, OSError) as e:
        logger.error(f"Error al guardar la comparación de modelos en {comparison_csv}: {e}", exc_info=True)
    
    # Guardar todas las predicciones
    predictions_df_data = {'Real': y_test.values} # Usar .values para evitar problemas de índice
    for model_key, (_, preds, _) in models.items():
        model_pred_col_name = f'Pred_{model_key.replace(" ", "_")}'
        predictions_df_data[model_pred_col_name] = preds
        
    predictions_df = pd.DataFrame(predictions_df_data, index=y_test.index) # Mantener índice original de y_test
    
    predictions_file = PROCESSED_DATA_DIR / 'all_model_predictions.csv'
    try:
        predictions_df.to_csv(predictions_file, index=True) # Guardar con índice para posible alineación posterior
        logger.info(f"Todas las predicciones de modelos guardadas en {predictions_file}")
    except (IOError, OSError) as e:
        logger.error(f"Error al guardar las predicciones en {predictions_file}: {e}", exc_info=True)
    
    logger.info("Predicción de rendimiento académico completada.")
    return models

if __name__ == "__main__":
    # Configuración básica de logging para pruebas
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("predictive_models_test.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.info("Ejecutando predictive_models.py como script principal para prueba.")
    # Importar módulo de carga de datos
    from src.data.data_loader import get_data # Movido aquí para que el logger de este módulo esté configurado
    
    # Cargar datos
    logger.info("Cargando datos para la prueba de modelos predictivos...")
    df = get_data()
    
    # Entrenar modelos predictivos
    if df is not None:
        logger.info("Datos cargados, procediendo con predict_academic_performance.")
        predict_academic_performance(df)
    else:
        logger.error("No se pudieron cargar los datos para la prueba de modelos predictivos.")
