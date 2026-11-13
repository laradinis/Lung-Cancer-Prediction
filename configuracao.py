"""
configuracao.py: importações, threads e funções gerais, partilhadas pelos três notebooks.
Tem de ser o PRIMEIRO import de cada notebook ("from configuracao import *"): a ordem dos imports
e as variáveis de ambiente das threads só funcionam se vierem antes de tudo o resto.
Para mudar o número de processos/threads, alterar N_JOBS aqui e reiniciar o kernel.
"""
# =============================================================================================
# 1. CONFIGURAÇÃO DO AMBIENTE
# A ordem desta célula importa: as variáveis de ambiente e o import do LightGBM/CatBoost têm de vir
# antes do PyTorch, senão o kernel morre no Mac (cada biblioteca traz a sua cópia do OpenMP).
# =============================================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"          # Mac: várias cópias do OpenMP (PyTorch, LightGBM, ...)
# Processos em paralelo (pyradiomics, segmentação) e threads de cada modelo. Fixado aqui, ANTES dos imports, para
# valer em todas as cópias do OpenMP (substitui o threadpool_limits, que no Mac pode fazer o kernel morrer).
# Se o kernel morrer por falta de memória durante as extrações, baixar para 2.
N_JOBS = max(1, min(6, (os.cpu_count() or 2) - 1))   # nº de núcleos − 1, no máximo 6
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(N_JOBS)                      # limite de threads de cada biblioteca numérica

# No Mac, o LightGBM e o CatBoost têm de ser importados ANTES do PyTorch (conflito entre cópias do OpenMP)
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# PyTorch: só é usado na CNN da secção 7
import torch
import torch.nn as nn
import torch.nn.functional as F

# O pylidc usa np.int, np.float e SafeConfigParser, que já não existem nas versões atuais:
# recriam-se aqui, antes de importar o pylidc
import numpy as np
import configparser
np.int, np.float = int, float                       # compatibilidade do pylidc
configparser.SafeConfigParser = configparser.ConfigParser

# Bibliotecas gerais
import ast, re, pickle, hashlib, logging, warnings
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from joblib import Parallel, delayed                # paralelização das extrações de features

# Imagem médica: leitura dos CTs (pylidc para o LIDC, SimpleITK para o LNDb) e features (pyradiomics)
import pylidc as pl
import SimpleITK as sitk
from radiomics import featureextractor
from scipy import stats
from scipy import ndimage as ndi                     # filtros e morfologia 3D (segmentação da secção 9)
from scipy.ndimage import zoom                       # reamostragem dos cubos para 1 mm
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes, mesh_surface_area

# scikit-learn e companhia: modelos, validação e métricas
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, ParameterSampler, cross_val_predict
from sklearn.metrics import (balanced_accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error,
                             recall_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler      # reamostragem das classes raras
from imblearn.pipeline import Pipeline as ImbPipeline
from mord import LogisticIT                                       # regressão logística ordinal

# Os avisos do scikit-learn e do pandas enchem o output sem acrescentar nada
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_num_threads(N_JOBS)                        # o PyTorch tem o seu próprio limite de threads


def ler_notas(v):
    """Lista de notas, venha como lista ou como texto lido de um CSV (ex.: '[np.int64(3), 4]')."""
    if isinstance(v, (list, tuple, np.ndarray)):    # já é uma lista (tabela ainda em memória)
        return [float(x) for x in v]
    # Lida de um CSV: é texto. Tira-se o "np.int64(...)" e converte-se o texto numa lista Python
    return [float(x) for x in ast.literal_eval(re.sub(r"np\.\w+\(([^)]*)\)", r"\1", str(v)))]
