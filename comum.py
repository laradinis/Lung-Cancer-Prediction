"""
comum.py: código partilhado pelos três notebooks (secção 5 do notebook 1 e comparação dos modelos).
Ao ser importado ("from comum import *"), lê data/dataset_final_esfera.csv (criado pelas secções 2 a 4 do
notebook 1), separa o holdout, cria os folds e define as ferramentas de validação, de gravação dos
resultados e de comparação dos modelos.
"""
from configuracao import *  # importações e threads (tem de vir primeiro)
from IPython.display import display  # num ficheiro .py o display não existe por defeito


# =============================================================================================
# 5.1 ANTES DO TREINO: aprendido só no treino de cada fold (evita fuga de informação)
# =============================================================================================

def remover_quase_constantes(X, limiar_freq=0.95):
    # Frequência do valor mais comum de cada feature; saem as que têm o mesmo valor em > 95% dos nódulos
    freq = X.apply(lambda c: c.value_counts(normalize=True, dropna=False).iloc[0]) #Para cada coluna, 
    #calcula a frequência relativa (em percentagem, de 0 a 1) de cada valor e selecciona o valor mais 
    #comum através de .iloc[0]. O parâmetro dropna=False garante que os valores em falta (NaN) 
    # também são contabilizados.
    return list(X.columns[freq < limiar_freq])


def remover_correlacionadas(X, y, limiar=0.90):
    corr = X.corr(method="spearman").abs().fillna(0) # correlação entre features
    #Calcula a matriz de correlação de Spearman (que lida bem com relações não-lineares) entre todas as features de X.
    # O .abs() usa valores absolutos (ignora se a correlação é positiva ou negativa) e fillna(0) substitui eventuais valores nulos por 0.
    # Ordem: da feature mais correlacionada com a classe para a menos correlacionada
    ordem = X.corrwith(y, method="spearman").abs().fillna(0).sort_values(ascending=False).index
    aceites = []
    for f in ordem:
        if (corr.loc[f, aceites] <= limiar).all(): # não é redundante com nenhuma já aceite
            aceites.append(f)
    return aceites


def _ajustar_preprocessamento(X, y, limiar_corr, limiar_freq, selecao):
    X = X.drop(columns=["mascara_recurso"], errors="ignore") #faz drop da máscara de resurso
    medianas = X.median() # medianas do TREINO
    Xi = X.fillna(medianas) # Preenche o valores Nan com a mediana mas em prinípio não há valores Nan
    #faz selecção de features se selecao = true
    f = remover_correlacionadas(Xi[remover_quase_constantes(Xi, limiar_freq)], y, limiar_corr) if selecao else list(X.columns)
    # O que é preciso para transformar qualquer conjunto da mesma forma: features, medianas e normalização
    return {"features": f, "medianas": medianas[f], "scaler": StandardScaler().fit(Xi[f])} 


_CACHE_PREP = {} # cache em memória
PASTA_CACHE_PREP = Path("cache_preprocessamento") # cache em disco


def ajustar_preprocessamento(X, y, limiar_corr=0.90, limiar_freq=0.95, selecao=True):
    """Medianas, features escolhidas e normalização, aprendidas no treino (com cache; selecao=False mantém todas)."""
    if not selecao: # sem seleção é rápido: não precisa de cache
        return _ajustar_preprocessamento(X, y, limiar_corr, limiar_freq, False)
    # Chave da cache = hash do conteúdo do treino (valores, labels, colunas e limiares)
    Xt = X.drop(columns=["mascara_recurso"], errors="ignore") #faz drop da mascara de recurso como na outra
    h = hashlib.md5() #calcula a impressão digital dos dados neste exato momento.
    h.update(np.ascontiguousarray(Xt.to_numpy(dtype=float)).tobytes())
    h.update(np.asarray(y).astype(np.int64).tobytes())
    h.update(("|".join(map(str, Xt.columns)) + f"|{limiar_corr}|{limiar_freq}").encode())
    chave = h.hexdigest()
    if chave not in _CACHE_PREP:
        ficheiro = PASTA_CACHE_PREP / f"{chave}.pkl"
        if ficheiro.exists():                                           # já calculado numa execução anterior
            _CACHE_PREP[chave] = pd.read_pickle(ficheiro)
        else:
            _CACHE_PREP[chave] = _ajustar_preprocessamento(X, y, limiar_corr, limiar_freq, True)
            PASTA_CACHE_PREP.mkdir(exist_ok=True)
            pd.to_pickle(_CACHE_PREP[chave], ficheiro)
    return _CACHE_PREP[chave]


def aplicar_preprocessamento(prep, X): #Usada no X teste
    # Aplica a outro conjunto (validação, holdout, teste) o que foi aprendido no treino
    f = prep["features"] #Vai buscar apenas a lista de colunas (features) que foram aprovadas e selecionadas no conjunto de treino
    return pd.DataFrame(prep["scaler"].transform(X[f].fillna(prep["medianas"])), columns=f, index=X.index)

# =============================================================================================
# 5.2 DADOS, HOLDOUT E PESOS
# =============================================================================================
df_total = pd.read_csv("data/dataset_final_esfera.csv")
# O raio da esfera lê-se do nome do ficheiro de features (não é preciso correr a secção 3 de novo)
RAIO_ESFERA = int(re.search(r"esfera(\d+)_", next(Path("data").glob("features_esfera*_teste.csv")).name).group(1))
#seleciona as features que podem estar no treino e teste
features = [c for c in df_total.columns if c not in ["idx", "dataset", "patient", "scan_id", "nodule",
                                                      "n_readers", "malignancy", "classe"]]
CLASSES = [1, 2, 3, 4, 5]

# Holdout: 1 dos 5 folds dos pacientes do LNDb
# (StratifiedGroupKFold: pacientes inteiros, proporções de classes semelhantes; seed fixa = sempre o mesmo holdout)
idx_lndb = np.where(df_total["dataset"] == "LNDb")[0] #vai buscar as linhas que correspondem ao lndb
#Aplica a divisão estratificada por grupos (mantendo pacientes inteiros juntos e equilibrando as classes) 
# especificamente sobre os dados do LNDb, extraindo os índices do primeiro fold (h) para servir de teste final.
_, h = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(
    idx_lndb, df_total["classe"].iloc[idx_lndb], groups=df_total["patient"].iloc[idx_lndb]))
e_holdout = np.zeros(len(df_total), dtype=bool) #Cria uma lista de valores False com o tamanho total do dataset, servindo de base para marcar quem vai pertencer ao holdout.
e_holdout[idx_lndb[h]] = True #Altera para True as posições exatas correspondentes aos pacientes do LNDb selecionados para o teste final.
df = df_total[~e_holdout].reset_index(drop=True) # Cria o conjunto de desenvolvimento (df), pegando em tudo o que não é holdout (~e_holdout) e reorganizando os índices.
df_holdout = df_total[e_holdout].reset_index(drop=True) #Cria o conjunto de holdout/teste final (df_holdout), isolando apenas as linhas marcadas como True.

# Variáveis usadas em todo o resto da primeira versão
X_dev, y, grupos = df[features], df["classe"], df["patient"]
estrato = df["dataset"] + "_" + df["classe"].astype(str) # estratificação pelo dataset de origem e classe
e_lndb = (df["dataset"] == "LNDb").to_numpy() #Cria uma máscara booleana para identificar rapidamente quais os registos do desenvolvimento que pertencem ao LNDb.

# Pesos pela concordância (2.4): lidos à parte, nunca são features
peso_lidc = pd.read_csv("data/nodulos.csv")["peso"] #Lê os pesos de concordância dos médicos para o dataset LIDC a partir do ficheiro CSV
peso_lndb = pd.read_csv("data/lndb_cubos.csv").set_index("idx")["peso"] #Lê os pesos para o LNDb e define a coluna "idx" como índice para facilitar buscas
pesos_total = np.where(df_total["dataset"] == "LIDC", peso_lidc.reindex(df_total["idx"]).to_numpy(),
                       peso_lndb.reindex(df_total["idx"]).to_numpy())
PESOS = pd.Series(pesos_total[~e_holdout], index=df.index)            # pesos do desenvolvimento
assert PESOS.notna().all(), "há nódulos sem peso: correr a secção 2.4"
print(f"Desenvolvimento: {len(df)} nódulos | holdout (LNDb): {len(df_holdout)}")



# =============================================================================================
# 5.3 FOLDS E MÉTRICAS
# =============================================================================================
SEEDS = [42, 7, 123] # 3 repetições × 5 folds = 15 treinos


def criar_splits(seed):
    # 5 folds: pacientes inteiros em cada fold, proporções de dataset × classe semelhantes
    return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X_dev, estrato, groups=grupos))


for seed in SEEDS:      
    #Percorre todas as seeds e folds criados e usa o operador & (interseção) para confirmar que nenhum paciente 
    # está simultaneamente no conjunto de treino (tr) e de validação (va). Se houvesse sobreposição, 
    # o código parava de imediato.                                             
    for tr, va in criar_splits(seed):
        assert not set(grupos.iloc[tr]) & set(grupos.iloc[va])


def metricas(y_verdadeiro, y_previsto):
    """As quatro métricas da competição, sempre com as 5 classes."""
    return {"bal_acc": balanced_accuracy_score(y_verdadeiro, y_previsto),
            "macro_f1": f1_score(y_verdadeiro, y_previsto, average="macro", labels=CLASSES, zero_division=0),
            "qwk": cohen_kappa_score(y_verdadeiro, y_previsto, weights="quadratic", labels=CLASSES),
            "mae": mean_absolute_error(y_verdadeiro, y_previsto)}


def resumo_lndb(oof):
    """Métricas no LNDb (previsões out-of-fold, uma coluna por seed): média e desvio entre seeds."""
    por_seed = pd.DataFrame([metricas(y[e_lndb], oof[s][e_lndb]) for s in oof.columns])
    return pd.DataFrame({"média": por_seed.mean(), "desvio": por_seed.std()})


# Pesos do QWK: (i − j)², a penalização cresce com o quadrado da distância entre notas
_W_V2 = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2).astype(float)


def metricas_rapidas(y_, p):
    """bal. acc., macro F1, QWK e MAE pela matriz de confusão (iguais às do scikit-learn, muito mais rápidas;
    usadas nos bootstraps)."""
    # M[i, j] = nº de nódulos da classe i+1 previstos como j+1
    M = np.bincount((np.asarray(y_, int) - 1) * 5 + (np.asarray(p, int) - 1), minlength=25).reshape(5, 5).astype(float)
    reais, prev, tp = M.sum(1), M.sum(0), np.diag(M) # totais reais, previstos, acertos por classe
    bal = (tp[reais > 0] / reais[reais > 0]).mean() # média do recall das classes presentes
    f1 = np.where(reais + prev > 0, 2 * tp / np.maximum(reais + prev, 1e-12), 0.0).mean()
    E = np.outer(reais, prev) / M.sum() # matriz esperada por acaso
    qwk = 1 - (_W_V2 * M).sum() / (_W_V2 * E).sum() if (_W_V2 * E).sum() > 0 else 0.0
    return bal, f1, qwk, (np.sqrt(_W_V2) * M).sum() / M.sum() # MAE: |i − j| médio


def recall_lndb(oof):
    # Recall de cada classe no LNDb, média entre seeds
    return np.mean([recall_score(y[e_lndb], oof[s][e_lndb], labels=CLASSES, average=None, zero_division=0)
                    for s in oof.columns], axis=0)


# =============================================================================================
# 5.4 VALIDAÇÃO CRUZADA E FERRAMENTAS DAS EXPERIÊNCIAS
# =============================================================================================
M4 = ["bal_acc", "macro_f1", "qwk", "mae"]
PASTA_CACHE_CV = Path("cache_avaliacoes") #Define a pasta no disco onde serão guardados os resultados em cache das avaliações.


def avaliar(criar, seeds=SEEDS, selecao=True, limiar_corr=0.90, X=None, chave=None):
    """
    Validação cruzada repetida. Devolve (métricas por fold, previsões out-of-fold com uma coluna por seed).
    criar(seed) devolve um modelo novo, que trata ele próprio dos pesos e do balanceamento (6.1), só no treino.
    chave: descrição da configuração; se for dada, o resultado fica em cache_avaliacoes/ e, ao voltar a correr
    o notebook, é lido do disco em vez de treinar tudo de novo (apagar a pasta para refazer).
    """
    # 1. Cache: se esta configuração já foi avaliada (com os mesmos dados), lê do disco
    caminho = None #Inicializa a variável do caminho do ficheiro de cache como vazia
    if chave is not None:
        h = hashlib.md5(repr((ASSINATURA, chave, list(seeds), selecao, limiar_corr)).encode()).hexdigest()[:16]
        caminho = PASTA_CACHE_CV / f"{h}.pkl"
        if caminho.exists():
            return pd.read_pickle(caminho)
    X = X_dev if X is None else X #Se não forem passados dados específicos, assume por defeito o conjunto de desenvolvimento (X_dev)
    # oof: uma linha por nódulo do desenvolvimento, uma coluna por seed (a nota prevista quando estava na validação)
    resultados, oof = [], pd.DataFrame(index=y.index, columns=seeds, dtype=float) #Inicializa uma lista vazia (resultados)
    #para guardar as métricas detalhadas de cada fold e um DataFrame do Pandas (oof) com uma linha por nódulo e uma coluna por seed para guardar as previsões out-of-fold.
    for seed in seeds:
        for fold, (tr, va) in enumerate(criar_splits(seed), start=1):
            # 2. Pré-processamento aprendido só no treino deste fold e aplicado à validação
            prep = ajustar_preprocessamento(X.iloc[tr], y.iloc[tr], limiar_corr, selecao=selecao)
            X_fit, X_val = aplicar_preprocessamento(prep, X.iloc[tr]), aplicar_preprocessamento(prep, X.iloc[va])
            # 3. Treino e previsão da validação, arredondada a uma nota de 1 a 5. O índice de X_fit identifica
            # os nódulos: é por ele que o modelo vai buscar os pesos e o alvo contínuo de cada um
            modelo = criar(seed).fit(X_fit, y.iloc[tr]) #instancia e treina o modelo
            y_pred = np.clip(np.round(modelo.predict(X_val)), 1, 5).astype(int) #Faz previsões no conjunto de validação 
            #(X_val), arredonda-as para o inteiro mais próximo, restringe os valores estritamente entre 1 e 5 com np.clip 
            # (já que as classes de malignidade vão de 1 a 5) e converte para tipo inteiro.
            oof.iloc[va, oof.columns.get_loc(seed)] = y_pred #Insere estas previsões nas linhas de validação exatas (va) 
            #e na coluna correspondente à seed atual dentro do DataFrame global OOF.
            resultados.append({"seed": seed, "fold": fold, "n_features": len(prep["features"]),
                               **metricas(y.iloc[va], y_pred)})
    saida = (pd.DataFrame(resultados), oof.astype(int)) #Converte a lista de resultados num DataFrame do Pandas e garante que as previsões OOF são do tipo inteiro, juntando ambos num tuplo (saida).
    if caminho is not None: # guarda para a próxima execução
        PASTA_CACHE_CV.mkdir(exist_ok=True)
        pd.to_pickle(saida, caminho) #guarda o tuplo de resultados em formato binário pickle na cache
    return saida


def tabela(oofs, recall=False):
    """Médias das 4 métricas e do S no LNDb (entre seeds) de vários modelos, ordenadas pelo S (o melhor primeiro);
    com recall=True, também o recall de cada classe. Devolve a tabela (usar t.loc[nome], não t.iloc)."""
    t = pd.DataFrame({n: resumo_lndb(o)["média"] for n, o in oofs.items()}).T[M4]
    t["S"] = t["bal_acc"] + t["macro_f1"] + t["qwk"] - t["mae"]          # S médio = média do S de cada seed
    t = t.sort_values("S", ascending=False)
    display(t.round(3))
    if recall:
        display(pd.DataFrame({n: recall_lndb(o) for n, o in oofs.items()},
                             index=[f"classe {c}" for c in CLASSES]).T.loc[t.index].round(2))
    return t


def ganha_em_3_de_4(novo, ref):
    """Para o modelo novo ser considerado melhor, tem de superar o anterior em pelo menos 3 das 4 métricas principais 
        (lembrando que no MAE, "melhor" significa ter um valor mais baixo).
        Devolve true se os pesos forem adotados"""
    # int(): com valores do numpy, True + True dá True (é um "ou" lógico) e a soma nunca chegaria a 3
    return sum(int(v) for v in (novo["bal_acc"] > ref["bal_acc"], novo["macro_f1"] > ref["macro_f1"],
                                novo["qwk"] > ref["qwk"], novo["mae"] < ref["mae"])) >= 3




def s_lndb(oof):
    """S = bal. acc. + macro F1 + QWK − MAE no LNDb, uma vez por seed (previsões out-of-fold)."""
    por_seed = pd.DataFrame([metricas(y[e_lndb], oof[s][e_lndb]) for s in oof.columns])
    return por_seed["bal_acc"] + por_seed["macro_f1"] + por_seed["qwk"] - por_seed["mae"]


def melhor_em_s(oof_novo, oof_ref):
    """Regra de um desvio: o novo só é melhor se o seu S superar o da referência por mais do que o desvio entre seeds."""
    s_n, s_r = s_lndb(oof_novo), s_lndb(oof_ref)
    return bool(s_n.mean() - s_r.mean() > max(s_n.std(), s_r.std()))


def _converter(v, opcoes):
    """Valor de uma tabela de resultados -> tipo original da grelha (None, str, int ou float)."""
    # O pandas guarda 300 como 300.0 e None como NaN: aqui volta-se ao tipo que o modelo espera
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return None
    numeros = [o for o in opcoes if o is not None and not isinstance(o, str)]
    return int(v) if all(isinstance(o, int) for o in numeros) else float(v)


def pesquisa(nome_cache, criar_cfg, grelha, seeds=SEEDS[:1], criterio="score", n_iter=20, **kw):
    """Pesquisa aleatória: criar_cfg(params, seed) -> modelo. Devolve os melhores hiperparâmetros."""
    caminho = Path("cache_tuning") / f"{nome_cache}.pkl" #Define um ficheiro para guardar os resultados desta pesquisa específica no disco.
    caminho.parent.mkdir(exist_ok=True)
    res = pd.read_pickle(caminho) if caminho.exists() else []  #Se a pesquisa já tiver sido interrompida e reiniciada 
    #antes, ele recupera o que já foi testado; caso contrário, começa com uma lista vazia.
    # ParameterSampler com seed fixa: as mesmas 20 combinações sempre, pela mesma ordem
    for i, p in enumerate(ParameterSampler(grelha, n_iter=n_iter, random_state=42)):
        if i < len(res): #Se esta combinação específica já tiver sido testada numa execução anterior
            continue #o código salta-a à frente para poupar tempo.
        _, oof = avaliar(lambda s, p=p: criar_cfg(p, s), seeds=seeds, **kw) #Avalia a combinação atual usando a validação cruzada.
        res.append({**p, **resumo_lndb(oof)["média"].to_dict()}) #Adiciona os hiperparâmetros testados e as respetivas métricas obtidas à lista de resultados.
        pd.to_pickle(res, caminho) # guarda depois de cada combinação
        # score = S = bal. acc. + macro F1 + QWK − MAE no LNDb (as quatro métricas da competição)
    t = pd.DataFrame(res).assign(score=lambda d: d["bal_acc"] + d["macro_f1"] + d["qwk"] - d["mae"]).sort_values(criterio, ascending=False)
    display(t.head(10).round(3))
    melhor = {k: _converter(t.iloc[0][k], v) for k, v in grelha.items()}
    print("Melhor combinação:", melhor) #Converte os valores da melhor linha para os tipos de dados corretos e imprime-os.
    return melhor #devolve a melhor combinação de hiperparâmetros



SEEDS_REAVALIACAO = [42, 123, 2024]



# =============================================================================================
# 5.5 (a) REGRESSÃO COM LIMIARES: prever a nota como número contínuo e convertê-la com 4 limiares
# =============================================================================================
# (o número de threads é fixado na secção 1, antes dos imports)

def qwk_rapido(y_true, y_pred, n=5):
    # QWK pela matriz de confusão (usado milhares de vezes na otimização dos limiares)
    O = np.zeros((n, n))
    np.add.at(O, (np.asarray(y_true, int) - 1, np.asarray(y_pred, int) - 1), 1)   # matriz observada
    W = np.subtract.outer(np.arange(n), np.arange(n)) ** 2                          # pesos (i − j)²
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()                                      # matriz esperada por acaso
    return 1 - (W * O).sum() / (W * E).sum()


def aplicar_limiares(pred, limiares): 
    # Converte as previsões contínuas do modelo em classes inteiras usando quatro limiares.
    """< t1 -> 1, [t1, t2[ -> 2, ..., ≥ t4 -> 5."""
    return np.digitize(pred, limiares) + 1


def otimizar_limiares(pred, y_true, passos=3): #Usada para os cotes ótimos
    """Ajusta os 4 limiares, um de cada vez, para maximizar o QWK."""
    # Começa no arredondamento normal (1,5 / 2,5 / 3,5 / 4,5) e testa valores de 0,02 em 0,02 (desde 1.0 até 5.0)
    lim, grelha = np.array([1.5, 2.5, 3.5, 4.5]), np.arange(1.0, 5.001, 0.02)
    melhor = qwk_rapido(y_true, aplicar_limiares(pred, lim))
    for _ in range(passos):
        for k in range(4):
            # O limiar k só pode mexer-se entre os vizinhos (os limiares ficam sempre por ordem)
            baixo, cima = (lim[k - 1] if k > 0 else 1.0), (lim[k + 1] if k < 3 else 5.0)
            for t in grelha[(grelha > baixo) & (grelha < cima)]:
                cand = lim.copy(); cand[k] = t
                q = qwk_rapido(y_true, aplicar_limiares(pred, cand))
                if q > melhor:
                    melhor, lim = q, cand
    return lim


class RegressaoComLimiares(BaseEstimator):
    """Regressor do alvo contínuo + limiares (fit / predict de notas 1 a 5). alvo, grupos, pesos: séries pelo índice de X."""
    def __init__(self, regressor, alvo, grupos, pesos=None, n_splits=5):
        self.regressor, self.alvo, self.grupos, self.pesos, self.n_splits = regressor, alvo, grupos, pesos, n_splits

    def fit(self, X, y):
        # O alvo contínuo (mediana das notas, ex.: 2,5), os pacientes e os pesos das linhas de X (pelo índice)
        y_cont, g = self.alvo.loc[X.index].to_numpy(), self.grupos.loc[X.index].to_numpy()
        w = None if self.pesos is None else self.pesos.loc[X.index].to_numpy()
        # Previsões out-of-fold DENTRO do treino (por paciente): os limiares não são escolhidos com
        # previsões de nódulos que o regressor já viu
        pred_oof = cross_val_predict(clone(self.regressor), X, y_cont, groups=g, cv=GroupKFold(self.n_splits),
                                     params=None if w is None else {"sample_weight": w})
        self.limiares_ = otimizar_limiares(pred_oof, np.asarray(y))
        # Modelo final: o mesmo regressor treinado com todo o treino
        self.modelo_ = clone(self.regressor).fit(X, y_cont, sample_weight=w)
        return self

    def predict(self, X):
        return aplicar_limiares(self.modelo_.predict(X), self.limiares_)

# =============================================================================================
# 5.5 (b) POPULAÇÕES DE TREINO, ASSINATURA DOS DADOS, TESTE E GRAVAÇÃO DOS RESULTADOS
# =============================================================================================

"""Em vez de termos de andar a passar várias variáveis soltas (como df, PESOS, X_dev, y, grupos) sempre que queremos
 treinar um modelo, a função _populacao junta tudo num único objeto (SimpleNamespace) para facilitar a chamada das
   funções."""
def _populacao(d, conc):
    # Tudo o que um modelo final precisa de uma população de treino
    return SimpleNamespace(d=d, X=d[features], y=d["classe"], alvo=d["malignancy"], grupos=d["patient"],
                           e_lndb=(d["dataset"] == "LNDb").to_numpy(), conc=conc)


POP_DEV = _populacao(df, PESOS)  # desenvolvimento (prevê o holdout)
POP_TUDO = _populacao(df_total, pd.Series(pesos_total, index=df_total.index)) # tudo (prevê o teste)


def pesos_de(pop, peso_lndb=1.0, concordancia=False):
    """Peso de cada nódulo: peso do LNDb x peso pela concordância (se concordancia=True)."""
    w = pd.Series(np.where(pop.e_lndb, peso_lndb, 1.0), index=pop.d.index)
    return w * pop.conc if concordancia else w

"""Atribui chaves (IDs) numéricas únicas a cada nódulo do dataset.

Estes offsets (100.000 para o holdout, 200.000 para o teste) servem para garantir que as imagens tridimensionais 
(cubos de TAC) associadas a cada nódulo nunca se sobrepõem e podem ser facilmente encontradas pelo sistema ao 
alimentar a CNN (Rede Neuronal Convolucional, Secção 7)."""
# Chaves das imagens (só para a CNN, secção 7): desenvolvimento = índice em df, holdout e teste com deslocamento
# (cada nódulo de qualquer conjunto tem uma chave única, para ir buscar a sua imagem)
OFFSET_HOLDOUT, OFFSET_TESTE = 100_000, 200_000
chave_total = np.empty(len(df_total), dtype=np.int64)
chave_total[~e_holdout] = df.index.to_numpy()
chave_total[e_holdout] = OFFSET_HOLDOUT + df_holdout.index.to_numpy()


def arredondar(pred):
    return np.clip(np.round(np.asarray(pred, dtype=float)), 1, 5).astype(int)

#Cria outra pasta de resultados para 
# Resultados em disco, com uma assinatura dos dados, do holdout, das seeds e das features
# (se algum destes mudar, os resultados antigos deixam de ser válidos e são ignorados)
PASTA_RESULTADOS = Path("resultados")
PASTA_RESULTADOS.mkdir(exist_ok=True)
_h = hashlib.md5()
_h.update(pd.util.hash_pandas_object(df_total[["idx", "dataset", "classe"]], index=False).to_numpy().tobytes())
_h.update(e_holdout.tobytes())
_h.update(repr((SEEDS, features)).encode())
ASSINATURA = _h.hexdigest()[:12]


def _caminho(nome):
    # "RF SMOTE" -> resultados/RF_SMOTE.pkl
    return PASTA_RESULTADOS / (re.sub(r"[^0-9A-Za-z]+", "_", nome).strip("_") + ".pkl")


def carregar_todos():
    """Modelos guardados em resultados/ com a assinatura atual (nome -> resultado), por ordem de gravação."""
    achados = []
    for c in PASTA_RESULTADOS.glob("*.pkl"):
        r = pd.read_pickle(c)
        if r.get("assinatura") == ASSINATURA:
            achados.append((c.stat().st_mtime, r)) # data de gravação, para ordenar
        else:
            print(f"Aviso: {c.name} foi guardado com outros dados, folds ou features e foi ignorado.")
    return {r["nome"]: r for _, r in sorted(achados, key=lambda t: t[0])} #ordena os modelos pela data de adição


_DADOS_TESTE = {}                                                        # lido uma só vez


def dados_teste():
    #Carrega e valida os dados do conjunto de teste final (lido apenas uma vez para poupar memória e tempo).
    """(achados do testNodules.csv, cubos do teste, features da esfera de cada cubo: linha i = cubo i)."""
    if not _DADOS_TESTE: #Se os dados ainda não estiverem carregados na memória (_DADOS_TESTE estiver vazio), executa o carregamento:
        cubos = pd.read_csv("data/teste_cubos.csv") #Lê os cubos de teste (teste_cubos.csv) 
        assert (cubos["idx"].to_numpy() == np.arange(len(cubos))).all(), "teste_cubos.csv desalinhado" #e valida se os índices estão alinhados
        feats = pd.read_csv(f"data/features_esfera{RAIO_ESFERA}_teste.csv").set_index("idx").sort_index() #Lê as features correspondentes ao raio da esfera (RAIO_ESFERA)
        X = cubos[["idx"]].join(feats, on="idx").drop(columns="idx") #junta as features
        faltam = [f for f in features if f not in X.columns] #Verifica se falta alguma feature obrigatória.
        assert not faltam, f"features em falta no teste: {faltam[:5]}"
        _DADOS_TESTE.update(teste=pd.read_csv("data/testNodules.csv"), cubos=cubos, X=X[features]) #Guarda tudo num dicionário global (_DADOS_TESTE).
    return _DADOS_TESTE["teste"], _DADOS_TESTE["cubos"], _DADOS_TESTE["X"]


def escrever_predictions(pred_cubos, caminho="predictions.csv", nota_por_defeito=2):
    """Ficheiro de submissão: as linhas e a ordem do testNodules.csv + 'prediction'; achados sem cubo -> nota 2."""
    teste, cubos, _ = dados_teste()
    assert len(pred_cubos) == len(cubos), "tem de haver uma previsão por cubo do teste"
    # Liga a previsão de cada cubo ao seu achado (LNDbID, FindingID), mantendo a ordem do testNodules.csv
    por_cubo = cubos[["LNDbID", "FindingID"]].assign(prediction=arredondar(pred_cubos))
    sub = teste.merge(por_cubo.drop_duplicates(["LNDbID", "FindingID"]), on=["LNDbID", "FindingID"], how="left")
    sem_previsao = int(sub["prediction"].isna().sum())                   # os 5 achados sem cubo
    sub["prediction"] = sub["prediction"].fillna(nota_por_defeito).astype(int)
    # Verificações do ficheiro entregue
    assert len(sub) == len(teste), "número de linhas diferente do testNodules.csv"
    assert (sub[["LNDbID", "FindingID"]].to_numpy() == teste[["LNDbID", "FindingID"]].to_numpy()).all(), \
        "ordem diferente do testNodules.csv"
    assert sub["prediction"].between(1, 5).all(), "há previsões fora de 1 a 5"
    sub.to_csv(caminho, index=False)
    print(f"{caminho}: {len(sub)} linhas | sem cubo (nota {nota_por_defeito}): {sem_previsao}")
    return sub


def finalizar_modelo(nome, criar, oof, chaves=False, descricao="", reutilizar=True, teste=True):
    """
    criar(pop) -> modelo novo para a população de treino. (1) treina no desenvolvimento e prevê o holdout;
    (2) se teste=True, treina com todos os nódulos e prevê o teste; grava tudo em resultados/. chaves=True para a CNN.
    Se o modelo já estiver guardado com os mesmos dados e a mesma configuração, é reutilizado (devolve None).
    """
    # 0. Já guardado com os mesmos dados e a mesma configuração? Então não treina de novo
    if reutilizar and _caminho(nome).exists():
        r = pd.read_pickle(_caminho(nome))
        if r.get("assinatura") == ASSINATURA and r.get("descricao") == descricao:
            print(f"{nome} | holdout: " +
                  " | ".join(f"{k} {v:.3f}" for k, v in metricas(df_holdout["classe"], r["pred_holdout"]).items()))
            return None
    # 1. Treino no desenvolvimento -> previsão do holdout
    prep = ajustar_preprocessamento(POP_DEV.X, POP_DEV.y)
    X_tr, X_h = aplicar_preprocessamento(prep, POP_DEV.X), aplicar_preprocessamento(prep, df_holdout[features])
    if chaves:                                                          # CNN: o índice é a chave da imagem
        X_h = X_h.set_axis(OFFSET_HOLDOUT + X_h.index)
    modelo_dev = criar(POP_DEV).fit(X_tr, POP_DEV.y)
    pred_holdout = arredondar(modelo_dev.predict(X_h))

    # 2. Treino com todos os nódulos (desenvolvimento + holdout) -> previsão do teste (só se teste=True)
    if not teste:
        pd.to_pickle({"nome": nome, "assinatura": ASSINATURA, "oof": oof, "pred_holdout": pred_holdout,
                      "pred_teste": None, "descricao": descricao}, _caminho(nome))
        print(f"{nome} | holdout: " + " | ".join(f"{k} {v:.3f}" for k, v in metricas(df_holdout["classe"], pred_holdout).items()))
        return modelo_dev
    _, cubos, X_teste = dados_teste()
    prep = ajustar_preprocessamento(POP_TUDO.X, POP_TUDO.y)
    X_all, y_all, X_t = aplicar_preprocessamento(prep, POP_TUDO.X), POP_TUDO.y, aplicar_preprocessamento(prep, X_teste)
    if chaves:
        X_all, y_all = X_all.set_axis(chave_total), y_all.set_axis(chave_total)
        X_t = X_t.set_axis(OFFSET_TESTE + cubos["idx"].to_numpy())
    pred_teste = arredondar(criar(POP_TUDO).fit(X_all, y_all).predict(X_t))

    # 3. Grava tudo (incluindo as previsões out-of-fold, usadas na comparação de cada notebook)
    pd.to_pickle({"nome": nome, "assinatura": ASSINATURA, "oof": oof, "pred_holdout": pred_holdout,
                  "pred_teste": pred_teste, "descricao": descricao}, _caminho(nome))
    print(f"{nome} | holdout: " + " | ".join(f"{k} {v:.3f}" for k, v in metricas(df_holdout["classe"], pred_holdout).items()))
    return modelo_dev

# =============================================================================================
# COMPARAÇÃO DOS MODELOS DE UM NOTEBOOK: LNDb (rank médio e bootstrap por paciente) e holdout (com IC)
# =============================================================================================

def juntar_notas(listas):
    # Ensemble da primeira versão: média das notas dos membros, arredondada (,5 para cima)
    return np.clip(np.floor(np.mean(listas, axis=0) + 0.5), 1, 5).astype(int)


def metr(yt, yp):
    return np.array(metricas_rapidas(yt, yp))                     # mesmos valores que metricas, muito mais rápido


def metr_boot(yt, yp):
    """Como metricas, mas as classes ausentes na amostra ficam fora do macro F1 (pela matriz de confusão: rápido)."""
    # Numa amostra do holdout pode não haver nenhum nódulo de classe 5: essa classe não conta para o macro F1
    M = np.bincount((yt - 1) * 5 + (yp - 1), minlength=25).reshape(5, 5).astype(float)
    reais, prev, tp = M.sum(1), M.sum(0), np.diag(M)
    bal, _, qwk, mae = metricas_rapidas(yt, yp)
    return np.array([bal, (2 * tp / np.maximum(reais + prev, 1e-12))[reais > 0].mean(), qwk, mae])


def amostras_bootstrap(pacientes, B, seed=42):
    """Bootstrap por paciente: sorteiam-se pacientes (com repetição) e entram todos os nódulos de cada um."""
    idx = {p: np.where(pacientes == p)[0] for p in np.unique(pacientes)}
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[p] for p in rng.choice(list(idx), len(idx))]) for _ in range(B)]


def tabela_ic(y_, preds, pacientes, B=1000, seed=42):
    """Valor [IC 95%] das 4 métricas de cada modelo {nome: notas}, por bootstrap de pacientes (mesmas amostras)."""
    amostras = amostras_bootstrap(pacientes, B, seed)
    linhas = {}
    for n, p in preds.items():
        boot = np.array([metr_boot(y_[i], p[i]) for i in amostras])
        pont = metricas(y_, p)
        linhas[n] = {m: f"{pont[m]:.3f} [{np.nanpercentile(boot[:, k], 2.5):.3f}, {np.nanpercentile(boot[:, k], 97.5):.3f}]"
                     for k, m in enumerate(M4)}
    return pd.DataFrame(linhas).T[M4]


def comparar_modelos(nomes, referencia=None, conjuntos=None, mostrar=True, regra="rank"):
    """
    Compara os modelos 'nomes' (guardados em resultados/) e os ensembles 'conjuntos' {nome: membros} no LNDb
    (previsões out-of-fold) e no holdout. Não escreve previsões do teste (só o notebook 3 gera o predictions.csv).
    Regra de decisão: o menor rank médio nas quatro métricas no LNDb (o critério do leaderboard da competição);
    em caso de empate, o de maior QWK. 'referencia' (opcional) é só o modelo face ao qual se mostram as diferenças.
    regra="3de4": em vez do rank, só se troca a 'referencia' por um modelo que a supere em pelo menos 3 das 4
    métricas no LNDb; se vários o fizerem, o que ganha em 3 de 4 a mais dos outros (empate: o de maior QWK).
    mostrar=False: só imprime a escolha (as tabelas e os gráficos ficam para o graficos.py).
    Devolve o nome do modelo escolhido.
    """
    todos = carregar_todos()
    em_falta = [n for n in nomes if n not in todos]
    if em_falta:
        print("Aviso: modelos ainda não guardados (ficam de fora):", em_falta)
    res = {n: todos[n] for n in nomes if n in todos}
    y_l = y.to_numpy()[e_lndb]                                            # labels do LNDb (desenvolvimento)

    # ---- LNDb: previsões out-of-fold de cada seed (modelos e ensembles) ----
    preds = {n: [r["oof"][s].to_numpy()[e_lndb] for s in r["oof"].columns] for n, r in res.items()}
    pred_h = {n: r["pred_holdout"] for n, r in res.items()}
    for nome, membros in (conjuntos or {}).items():
        if all(m in res for m in membros):
            n_seeds = min(len(preds[m]) for m in membros)
            preds[nome] = [juntar_notas([preds[m][s] for m in membros]) for s in range(n_seeds)]
            pred_h[nome] = juntar_notas([res[m]["pred_holdout"] for m in membros])
    P = pd.DataFrame({n: np.mean([metr(y_l, p) for p in ps], axis=0) for n, ps in preds.items()}, index=M4).T

    # ---- Regra de decisão: rank médio nas 4 métricas (no MAE, menor é melhor) ----
    rank = pd.concat([P[["bal_acc", "macro_f1", "qwk"]].rank(ascending=False), P[["mae"]].rank()], axis=1).mean(axis=1)
    if regra == "3de4" and referencia in P.index:
        # Regra dos 3 em 4: quem supera a referência em pelo menos 3 das 4 métricas no LNDb
        venc = [n for n in P.index if n != referencia and ganha_em_3_de_4(P.loc[n], P.loc[referencia])]
        vitorias = {n: sum(ganha_em_3_de_4(P.loc[n], P.loc[o]) for o in venc if o != n) for n in venc}
        escolha = max(venc, key=lambda n: (vitorias[n], P.loc[n, "qwk"])) if venc else referencia
        print(f"Superam {referencia} em 3 de 4 métricas no LNDb: {venc or 'nenhum'} | ESCOLHIDO: {escolha}")
    else:
        escolha = min(rank.index, key=lambda n: (rank[n], -P.loc[n, "qwk"]))
        print("Rank médio no LNDb (menor é melhor): " +
              " | ".join(f"{n} {v:.2f}" for n, v in rank.sort_values().items()) + f" | ESCOLHIDO: {escolha}")
    if not mostrar:
        return escolha

    # ---- Tabelas: LNDb com IC (bootstrap por paciente) e diferença face à referência; holdout com IC ----
    amostras = amostras_bootstrap(grupos.to_numpy()[e_lndb], 500)         # as mesmas amostras para todos
    boot = {n: np.array([np.mean([metr(y_l[i], p[i]) for p in ps], axis=0) for i in amostras]) for n, ps in preds.items()}
    ref = referencia if referencia in preds else escolha
    linhas = {}
    for n in preds:
        lo, hi = np.percentile(boot[n], [2.5, 97.5], axis=0)
        d_lo, d_hi = np.percentile(boot[n] - boot[ref], [2.5, 97.5], axis=0)
        d = P.loc[n].to_numpy() - P.loc[ref].to_numpy()
        linhas[n] = [f"{P.loc[n].iloc[k]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}] | {d[k]:+.3f}" +
                     (" *" if n != ref and (d_lo[k] > 0 or d_hi[k] < 0) else "") for k in range(4)]
    print(f"LNDb: valor [IC 95%] | diferença face a {ref} (* = o IC da diferença exclui o zero)")
    display(pd.DataFrame(linhas, index=M4).T.assign(rank=rank.round(2)))
    y_h = df_holdout["classe"].to_numpy()
    print(f"Holdout: {len(y_h)} nódulos | classes: {np.bincount(y_h, minlength=6)[1:].tolist()} | valor [IC 95%]:")
    display(tabela_ic(y_h, pred_h, df_holdout["patient"].to_numpy()))
    return escolha