"""
graficos.py: tabelas e gráficos dos três notebooks. Não treina nada: só mostra resultados já calculados.
- Tabelas: devolvem objetos Styler (ou listas deles) e são mostradas com mostrar(...), que as põe lado a lado.
  Estilo simples, legível em tema claro e escuro: o melhor valor de cada coluna (no MAE, o mais baixo) a negrito
  e a roxo; a linha (ou célula) escolhida com um fundo roxo claro; intervalos e desvios a cinzento, mais pequenos.
- Resultados da validação cruzada dos notebooks 1 e 2: dicionários {nome: (métricas por fold, oof)}, como os que
  avaliar() devolve. Tabelas da secção 11 do notebook 3: as de experiencia() e media_modelos().
- Tamanho dos gráficos: os gráficos só do notebook 3 são sempre compactos. Os gráficos partilhados (matrizes,
  recall, distribuição e importâncias) só ficam compactos com COMPACTO = True, que o notebook 3 liga logo a seguir
  ao import (G.COMPACTO = True); nos notebooks 1 e 2 ficam como estavam.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display, HTML

import comum as C # o mesmo módulo já importado pelo notebook (não volta a ler os dados)

M4 = ["bal_acc", "macro_f1", "qwk", "mae"]
NOMES = {"bal_acc": "Bal. acc.", "macro_f1": "Macro F1", "qwk": "QWK", "mae": "MAE", "S": "S"}
ROXO, ROXO_ESCURO, CINZA = "#be8aee", "#5b2a86", "#c8c8c8"
CORES = ["#5b2a86", "#9b59d0", "#c9a3ef", "#8c8c8c", "#3f3f3f", "#e0cdf6"] # roxos e cinzentos, por ordem
_BAL = {"None": "sem balanceamento", "classe": "pesos de classe", "ros": "RandomOverSampler", "smote": "SMOTE"}

# Gráficos partilhados mais pequenos (o notebook 3 liga isto com G.COMPACTO = True)
COMPACTO = False


def _tam(normal, compacto):
    """Tamanho de um gráfico partilhado: o normal (notebooks 1 e 2) ou o compacto (notebook 3)."""
    return compacto if COMPACTO else normal


# Estilo das tabelas: cores transparentes, para funcionar com o tema claro e com o escuro
_MELHOR = "font-weight: 700; color: #9b59d0"  # melhor valor
_DESTAQUE = "background-color: rgba(190, 138, 238, 0.22)" # linha/célula escolhida
_CSS = [{"selector": "", "props": "border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; "
                                  "margin: 0"},
        {"selector": "caption", "props": "caption-side: top; text-align: left; font-size: 12px; font-weight: 600; "
                                         "padding: 0 0 6px 2px"},
        {"selector": "th", "props": "padding: 5px 10px; font-weight: 600; text-align: right; white-space: nowrap; "
                                    "border-bottom: 1px solid rgba(128, 128, 128, 0.55)"},
        {"selector": "th.row_heading", "props": "text-align: left; font-weight: 400; "
                                                "border-bottom: 1px solid rgba(128, 128, 128, 0.18)"},
        {"selector": "th.col_heading.level0", "props": "text-align: center"},
        {"selector": "td", "props": "padding: 5px 10px; text-align: right; white-space: nowrap; "
                                    "border-bottom: 1px solid rgba(128, 128, 128, 0.18)"}]


def _cinza(texto):
    return f'<span style="opacity: 0.6; font-size: 10.5px">{texto}</span>'


def _ic(v, lo, hi):
    """Valor com o intervalo por baixo, mais pequeno e a cinzento."""
    return f"{v:.3f}<br>{_cinza(f'{lo:.2f} - {hi:.2f}')}"


def _dp(v, d, casas=3):
    """Média ± desvio, com o desvio a cinzento."""
    return f"{v:.3f} {_cinza(f'± {d:.{casas}f}')}"


def legivel(rotulo):
    """'pesos sim | bal classe | LNDb x4 | afinada' → 'pesos pela concordância, pesos de classe, LNDb x4, ...'."""
    if not isinstance(rotulo, str) or "|" not in rotulo:
        return rotulo
    trad = lambda p: ("pesos pela concordância" if p.endswith("sim") else "sem pesos pela concordância") \
        if p.startswith("pesos ") else _BAL.get(p[4:], p[4:]) if p.startswith("bal ") \
        else "hiperparâmetros da pesquisa" if p == "afinada" else p
    return ", ".join(trad(p.strip()) for p in rotulo.split("|"))


# =============================================================================================
# Tabelas
# =============================================================================================

def _menor_melhor(coluna):
    c = coluna[-1] if isinstance(coluna, tuple) else coluna
    return str(c).split(" ")[0].upper() == "MAE"


def _formatar(df, casas=3):
    """Números → texto com 'casas' decimais (o resto fica igual)."""
    f = lambda x: (f"{x:.{casas}f}" if isinstance(x, (float, np.floating)) else
                   f"{x}" if isinstance(x, (int, np.integer)) and not isinstance(x, bool) else x)
    return df.map(f) if hasattr(df, "map") else df.applymap(f)


def estilo(texto, num=None, legenda=None, melhores=None, por_linha=False, linhas=None, celulas=None):
    """Tabela (Styler). num: valores numéricos, para marcar o melhor de cada coluna (ou de cada linha, se
    por_linha=True); melhores: {coluna: linha} já calculados; linhas: linhas a destacar; celulas: {(linha, coluna)}."""
    est = texto.style
    if num is not None:
        cols = [c for c in texto.columns if c in num.columns]
        if por_linha:
            def _negrito(linha):
                v = num.loc[linha.name, cols].astype(float)
                return [_MELHOR if v.notna().any() and c == v.idxmax() else "" for c in cols]
            est = est.apply(_negrito, axis=1, subset=cols)
        else:
            for c in cols:
                v = num[c].astype(float)
                alvo = (melhores or {}).get(c, (v.idxmin() if _menor_melhor(c) else v.idxmax())
                                            if v.notna().any() else None)
                est = est.apply(lambda s, a=alvo: [_MELHOR if i == a else "" for i in s.index], subset=[c], axis=0)
    if linhas:
        est = est.apply(lambda s: [_DESTAQUE if s.name in linhas else "" for _ in s], axis=1)
        est = est.apply_index(lambda s: [_DESTAQUE if v in linhas else "" for v in s], axis=0)
    if celulas:
        est = est.apply(lambda s: [_DESTAQUE if (s.name, c) in celulas else "" for c in s.index], axis=1)
    est = est.set_table_styles(_CSS)
    return est.set_caption(legenda) if legenda else est


def mostrar(*tabelas):
    """Mostra várias tabelas (Styler, DataFrame ou listas deles) lado a lado, alinhadas pelo topo."""
    planas = [t for x in tabelas for t in (x if isinstance(x, (list, tuple)) else [x])]
    try:
        html = "".join(f'<div style="margin: 0 32px 16px 0">'
                       f'{(t.style.set_table_styles(_CSS) if isinstance(t, pd.DataFrame) else t).to_html()}</div>'
                       for t in planas)
        display(HTML(f'<div style="display: flex; flex-wrap: wrap; align-items: flex-start">{html}</div>'))
    except Exception: # sem jinja2: uma de cada vez
        for t in planas:
            display(t)


def _partes(texto, num, legenda, max_linhas, linhas=None):
    """Tabela inteira, ou partida em duas metades (lado a lado) se for comprida; o melhor é o da tabela toda."""
    mel = None if num is None else {c: (num[c].idxmin() if _menor_melhor(c) else num[c].idxmax())
                                    for c in num if num[c].notna().any()}
    n = len(texto)
    cortes = [slice(0, n)] if n <= max_linhas else [slice(0, (n + 1) // 2), slice((n + 1) // 2, n)]
    return [estilo(texto.iloc[p], None if num is None else num.iloc[p], legenda if k == 0 else None, mel,
                   linhas=linhas) for k, p in enumerate(cortes)]


def tabela_compacta(df, legenda=None, casas=3, colorir=True, cores=None, max_linhas=12, por_linha=False):
    """Qualquer tabela. colorir: marca o melhor valor das colunas numéricas (ou só das colunas da lista 'cores');
    por_linha=True: marca o melhor de cada linha (entre as colunas 'cores')."""
    num = None
    if colorir:
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num[[c for c in (cores or num.columns) if num[c].notna().any()]]
    if por_linha:
        return [estilo(_formatar(df.copy(), casas), num, legenda, por_linha=True)]
    return _partes(_formatar(df.copy(), casas), num, legenda, max_linhas)


def tabela_metricas(linhas, legenda=None, casas=3):
    """Uma linha por conjunto {nome: valores}, com as 4 métricas (e o S, se houver 5 valores). Cada valor pode ser:
    números; (valor, inferior, superior) → intervalo por baixo; (média, desvio) → média ± desvio."""
    texto, num = {}, {}
    for nome, v in linhas.items():
        if isinstance(v, dict):
            v = [v[m] for m in M4 + (["S"] if "S" in v else [])]
        if isinstance(v, tuple) and len(v) == 3:                      # (valor, inferior, superior)
            num[nome] = list(v[0])
            texto[nome] = [_ic(a, b, c) for a, b, c in zip(*v)]
        elif isinstance(v, tuple) and len(v) == 2:                    # (média, desvio)
            num[nome] = list(v[0])
            texto[nome] = [_dp(a, b) for a, b in zip(*v)]
        else:
            num[nome] = list(v)
            texto[nome] = [f"{a:.{casas}f}" for a in v]
    cols = [NOMES[m] for m in (M4 + ["S"])[:max(len(x) for x in num.values())]]
    texto = pd.DataFrame(texto, index=cols).T
    num = pd.DataFrame(num, index=cols).T
    return [estilo(texto, num if len(num) > 1 else None, legenda)]


def tabela_confusao(yt, yp, legenda=None):
    """Matriz de confusão em tabela (linhas: nota real; colunas: nota prevista); os acertos a negrito."""
    M = C.confusion_matrix(yt, yp, labels=C.CLASSES)
    df = pd.DataFrame(M, index=[f"Real {c}" for c in C.CLASSES], columns=[f"Prev. {c}" for c in C.CLASSES])
    est = df.style.apply(lambda s: [_MELHOR if s.index.get_loc(c) == df.index.get_loc(s.name) else
                               ("opacity: 0.45" if s[c] == 0 else "") for c in s.index], axis=1)
    est = est.set_table_styles(_CSS)
    return [est.set_caption(legenda) if legenda else est]


def tabela_distribuicao(conjuntos, legenda=None):
    """% de cada nota em cada conjunto {nome: notas}; uma coluna por conjunto."""
    tab = pd.DataFrame({n: pd.Series(np.asarray(v)).value_counts(normalize=True) for n, v in conjuntos.items()})
    tab = (tab.reindex(C.CLASSES).fillna(0) * 100).set_axis([f"Nota {c}" for c in C.CLASSES])
    est = tab.style.format("{:.0f}%").set_table_styles(_CSS)
    return [est.set_caption(legenda) if legenda else est]


def tabela_s(t, legenda=None, escolhido=None, max_linhas=10):
    """Tabela de experiencia()/media_modelos() (notebook 3): métricas no LNDb e S ± desvio; a escolhida destacada."""
    num = t[M4 + ["S"]].astype(float).rename(columns=NOMES)
    texto = _formatar(num.copy())
    texto["S"] = [_dp(a, b) for a, b in zip(t["S"], t["S_desvio"])]
    texto.index = num.index = [str(i) for i in t.index]
    return _partes(texto, num, legenda, max_linhas, linhas={str(escolhido)})


def tabela_por_modelo(t, legenda=None, escolhas=None, media=None, curto=None, ordem=None):
    """Tabela de experiencia() com linhas 'Modelo | opção' -> uma linha por modelo e uma coluna por opção (S ± desvio).
    escolhas: {modelo: opção} ou uma opção comum a todos (célula destacada). media: tabela de media_modelos()."""
    curto = curto or (lambda o: o)
    pares = [str(i).split(" | ", 1) for i in t.index]
    modelos, opcoes = list(dict.fromkeys(a for a, _ in pares)), list(dict.fromkeys(b for _, b in pares))
    opcoes = [o for o in (ordem or []) if o in opcoes] + [o for o in opcoes if o not in (ordem or [])]
    linhas = [(a, b, t.iloc[k]) for k, (a, b) in enumerate(pares)]
    if media is not None:
        modelos.append("Média dos modelos")
        linhas += [("Média dos modelos", b, media.loc[b]) for b in opcoes if b in media.index]
    num = pd.DataFrame(np.nan, index=modelos, columns=[curto(o) for o in opcoes])
    texto = pd.DataFrame("", index=modelos, columns=num.columns)
    escolhidas = set()
    for a, b, r in linhas:
        if not isinstance(escolhas, str) and (escolhas or {}).get(a) == b:
            escolhidas.add((a, curto(b)))
        num.loc[a, curto(b)] = r["S"]
        texto.loc[a, curto(b)] = _dp(r["S"], r["S_desvio"], 2)
    est = estilo(texto, num, legenda, por_linha=True, celulas=escolhidas)
    if isinstance(escolhas, str): # decisão comum: destaca a coluna escolhida
        est = est.apply(lambda s: [_DESTAQUE] * len(s), subset=[curto(escolhas)], axis=0)
        est = est.apply_index(lambda s: [_DESTAQUE if v == curto(escolhas) else "" for v in s], axis=1)
    if media is not None: # separa a linha da média
        est = est.set_table_styles(_CSS + [{"selector": "tbody tr:last-child td, tbody tr:last-child th",
                                            "props": "border-top: 1px solid rgba(128, 128, 128, 0.55)"}])
    return est


def _resumo(resultados):
    """{chave: (res, oof)} → (médias, desvios, todos os nódulos), uma linha por chave (notebooks 1 e 2)."""
    med, dp, todos = {}, {}, {}
    for k, (res, oof) in resultados.items():
        r = C.resumo_lndb(oof)
        med[k], dp[k], todos[k] = r["média"][M4], r["desvio"][M4], res[M4].mean()
    med, dp, todos = pd.DataFrame(med).T, pd.DataFrame(dp).T, pd.DataFrame(todos).T
    if isinstance(med.index[0], tuple):
        idx = pd.MultiIndex.from_tuples([(a, legivel(b)) for a, b in med.index])
        med.index = dp.index = todos.index = idx
    return med, dp, todos


def _s_por_seed(oof):
    """S = bal. acc. + macro F1 + QWK - MAE no LNDb, uma vez por seed (previsões out-of-fold)."""
    y_l = C.y.to_numpy()[C.e_lndb]
    return np.array([np.dot(C.metricas_rapidas(y_l, oof[s].to_numpy()[C.e_lndb]), [1, 1, 1, -1]) for s in oof.columns])


def _ordem_s(S, index):
    """Posições por S decrescente; com (modelo, versão), os modelos ficam juntos (o de melhor S primeiro)."""
    S = np.asarray(S, dtype=float)
    if isinstance(index, pd.MultiIndex):
        mod = np.array([k[0] for k in index])
        melhor = {m: S[mod == m].max() for m in set(mod)}
        return sorted(range(len(S)), key=lambda i: (-melhor[mod[i]], mod[i], -S[i]))
    return list(np.argsort(-S, kind="stable"))


def tabela_bonita(resultados, recall=False, todos_nodulos=False, legenda=None, ordenar=True):
    """Resultados da validação cruzada: LNDb (média ± desvio entre seeds, incluindo o S) e o QWK (ou as 4 métricas)
    em todos os nódulos; ordenada pelo S no LNDb (ordenar=False mantém a ordem). recall=True: recall de cada nota."""
    med, dp, todos = _resumo(resultados)
    s_seeds = [_s_por_seed(oof) for _, oof in resultados.values()]
    med["S"], dp["S"] = [s.mean() for s in s_seeds], [s.std(ddof=1) for s in s_seeds]
    o = _ordem_s(med["S"], med.index) if ordenar else list(range(len(med)))
    med, dp, todos = med.iloc[o], dp.iloc[o], todos.iloc[o]
    g1, g2 = "LNDb (média ± desvio entre seeds)", "Todos os nódulos"
    texto = pd.DataFrame({(g1, NOMES[m]): [_dp(a, b) for a, b in zip(med[m], dp[m])] for m in M4 + ["S"]},
                         index=med.index)
    num = pd.DataFrame({(g1, NOMES[m]): med[m].to_numpy() for m in M4 + ["S"]}, index=med.index)
    for m in (M4 if todos_nodulos else ["qwk"]):
        texto[(g2, NOMES[m])], num[(g2, NOMES[m])] = [f"{x:.3f}" for x in todos[m]], todos[m].to_numpy()
    texto.columns = num.columns = pd.MultiIndex.from_tuples(texto.columns)
    tabs = _partes(texto, num, legenda or "Validação cruzada, ordenada pelo S no LNDb", 99)
    if recall:
        rec = pd.DataFrame([C.recall_lndb(o_) for _, o_ in resultados.values()],
                           columns=[f"Nota {c}" for c in C.CLASSES]).iloc[o].set_axis(med.index)
        tabs += tabela_compacta(rec, "Recall de cada nota no LNDb", casas=2, max_linhas=99)
    mostrar(tabs)


def resumo_passo(resultados, escolhidas, titulo):
    """Uma linha por modelo: a versão escolhida neste passo e as 4 métricas e o S no LNDb (média ± desvio),
    ordenada pelo S."""
    sel = {m: k for m, k in escolhidas.items() if k in resultados}
    med, dp, _ = _resumo({m: resultados[k] for m, k in sel.items()})
    s_seeds = [_s_por_seed(resultados[k][1]) for k in sel.values()]
    med["S"], dp["S"] = [x.mean() for x in s_seeds], [x.std(ddof=1) for x in s_seeds]
    med.index = dp.index = list(sel)
    o = _ordem_s(med["S"], med.index)
    med, dp = med.iloc[o], dp.iloc[o]
    texto = pd.DataFrame({"Versão escolhida": [legivel(sel[m][1]) if isinstance(sel[m], tuple) else "" for m in med.index]},
                         index=med.index)
    for m in M4 + ["S"]:
        texto[NOMES[m]] = [_dp(a, b) for a, b in zip(med[m], dp[m])]
    est = _partes(texto, med.rename(columns=NOMES), f"{titulo}: LNDb, média ± desvio entre seeds", 99)
    mostrar([e.set_properties(subset=["Versão escolhida"], **{"text-align": "left"}) for e in est])


_CACHE_COMP = {}


def dados_comparacao(nomes, conjuntos=None, B=500):
    """LNDb (valor e IC 95% por bootstrap de pacientes) e holdout, para modelos gravados e ensembles."""
    todos = C.carregar_todos()
    # A chave inclui a data de gravação de cada modelo: se um modelo for gravado (ou regravado) depois, recalcula
    gravados = tuple((n, C._caminho(n).stat().st_mtime) for n in nomes if n in todos)
    chave = (tuple(nomes), repr(conjuntos), B, gravados)
    if chave not in _CACHE_COMP:
        res = {n: todos[n] for n in nomes if n in todos}
        y_l, y_h = C.y.to_numpy()[C.e_lndb], C.df_holdout["classe"].to_numpy()
        preds = {n: [r["oof"][s].to_numpy()[C.e_lndb] for s in r["oof"].columns] for n, r in res.items()}
        hold = {n: r["pred_holdout"] for n, r in res.items()}
        for nome, membros in (conjuntos or {}).items():
            if all(m in res for m in membros):
                k = min(len(preds[m]) for m in membros)
                preds[nome] = [C.juntar_notas([preds[m][s] for m in membros]) for s in range(k)]
                hold[nome] = C.juntar_notas([res[m]["pred_holdout"] for m in membros])
        amostras = C.amostras_bootstrap(C.grupos.to_numpy()[C.e_lndb], B)
        n_ = list(preds)
        boot = np.array([[np.mean([C.metr(y_l[i], p[i]) for p in preds[n]], axis=0) for i in amostras] for n in n_])
        _CACHE_COMP[chave] = dict(nomes=n_, pont=np.array([np.mean([C.metr(y_l, p) for p in preds[n]], axis=0) for n in n_]),
                                  lo=np.percentile(boot, 2.5, axis=1), hi=np.percentile(boot, 97.5, axis=1),
                                  holdout=np.array([C.metr(y_h, hold[n]) for n in n_]), preds=preds, hold=hold)
    return _CACHE_COMP[chave]


def tabela_comparacao(nomes, conjuntos=None, escolhido=None, B=500):
    """Tabela final: LNDb (valor e IC 95%, e S) e holdout de cada modelo e ensemble, ordenada pelo S no LNDb;
    o escolhido destacado."""
    d = dados_comparacao(nomes, conjuntos, B)
    S = d["pont"] @ np.array([1, 1, 1, -1])
    o = np.argsort(-S, kind="stable")
    rot = [d["nomes"][i] for i in o]
    g1, g2 = "LNDb (valor e IC 95%)", "Holdout"
    cols = pd.MultiIndex.from_tuples([(g1, NOMES[m]) for m in M4 + ["S"]] + [(g2, NOMES[m]) for m in M4])
    num = pd.DataFrame(np.c_[d["pont"][o], S[o], d["holdout"][o]], index=rot, columns=cols)
    texto = _formatar(num.copy())
    for j, m in enumerate(M4):
        texto[(g1, NOMES[m])] = [_ic(p, a, b) for p, a, b in zip(d["pont"][o, j], d["lo"][o, j], d["hi"][o, j])]
    legenda = "Comparação no LNDb (IC 95% por bootstrap de pacientes) e no holdout, ordenada pelo S no LNDb" + \
              (" (a destacado: o escolhido)" if escolhido is not None else "")
    mostrar(_partes(texto, num, legenda, 99, linhas={escolhido}))


# =============================================================================================
# Gráficos
# =============================================================================================

def _limpar(fig):
    """Estilo comum: sem as bordas de cima e da direita, eixos e grelha discretos."""
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9a9a9a")
        ax.tick_params(colors="#555555", labelsize=8)
        ax.title.set_fontsize(10)


def _fim(fig=None, titulo=None, rect=None):
    fig = fig or plt.gcf()
    _limpar(fig)
    if titulo:
        fig.suptitle(titulo, fontsize=11)
    plt.tight_layout(rect=rect) if rect else plt.tight_layout()
    plt.show()


def grafico_barras(resultados, titulo):
    """4 painéis: barras = LNDb (média ± desvio entre seeds, a melhor a roxo); losangos = todos os nódulos."""
    med, dp, todos = _resumo(resultados)
    nomes = [", ".join(k) if isinstance(k, tuple) else str(k) for k in med.index]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, m in zip(axes, M4):
        melhor = int(np.argmin(med[m]) if m == "mae" else np.argmax(med[m]))
        ax.bar(nomes, med[m], yerr=dp[m], color=[ROXO if i == melhor else CINZA for i in range(len(nomes))], capsize=3)
        ax.scatter(nomes, todos[m], marker="D", color=ROXO_ESCURO, zorder=3, label="todos os nódulos (LIDC + LNDb)")
        ax.set_ylim(0, max(med[m].max() + dp[m].max(), todos[m].max()) * 1.15)
        ax.set_title(NOMES[m] + (" (menor é melhor)" if m == "mae" else ""))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    _fim(fig, titulo)


def grafico_intervalos(blocos, titulo, losangos=None, referencia=None, teto=None):
    """Um painel por métrica: ponto e intervalo de cada bloco {nome: (valor, inferior, superior)}.
    losangos: {nome: valores} (ex.: holdout); referencia: linha vertical no valor desse bloco; teto: {métrica: valor}."""
    nomes = list(blocos)
    fig, axes = plt.subplots(1, 4, figsize=(16, 0.5 * len(nomes) + 1.9), sharey=True)
    ypos = np.arange(len(nomes))[::-1]
    for j, (ax, m) in enumerate(zip(axes, M4)):
        vals = [blocos[n][0][j] for n in nomes]
        melhor = int(np.argmin(vals) if m == "mae" else np.argmax(vals))
        for i, n in enumerate(nomes):
            v, lo, hi = blocos[n]
            c = ROXO_ESCURO if i == melhor else "#7f7f7f"
            ax.plot([lo[j], hi[j]], [ypos[i]] * 2, color=c, lw=2.5, alpha=0.8)
            ax.scatter(v[j], ypos[i], color=c, s=45, zorder=3)
            if losangos is not None and n in losangos:
                ax.scatter(losangos[n][j], ypos[i], marker="D", facecolor="none", edgecolor=ROXO, s=45, zorder=3)
        if referencia in blocos:
            ax.axvline(blocos[referencia][0][j], color="black", ls="--", lw=0.8)
        if teto is not None:
            ax.axvline(teto[m], color="black", ls=":", lw=1.2)
        ax.set_title(NOMES[m] + (" (menor é melhor)" if m == "mae" else ""))
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(nomes, fontsize=8)
    leg = [plt.Line2D([], [], color="#7f7f7f", lw=2.5, marker="o", label="valor e intervalo")]
    if losangos is not None:
        leg.append(plt.Line2D([], [], marker="D", ls="", mfc="none", mec=ROXO, label="holdout"))
    if referencia in blocos:
        leg.append(plt.Line2D([], [], color="black", ls="--", label=f"referência ({referencia})"))
    if teto is not None:
        leg.append(plt.Line2D([], [], color="black", ls=":", label="teto (radiologista vs. colegas)"))
    fig.legend(handles=leg, loc="lower center", ncol=len(leg), fontsize=8, frameon=False)
    _fim(fig, titulo, rect=(0, 0.08, 1, 1))


def grafico_comparacao(nomes, conjuntos=None, referencia=None, B=500):
    """Notebooks 1 e 2: LNDb (IC 95% por bootstrap de pacientes) de cada modelo gravado, com o holdout em losango."""
    d = dados_comparacao(nomes, conjuntos, B)
    blocos = {n: (d["pont"][i], d["lo"][i], d["hi"][i]) for i, n in enumerate(d["nomes"])}
    grafico_intervalos(blocos, "Comparação: LNDb (IC 95%) e holdout",
                       losangos={n: d["holdout"][i] for i, n in enumerate(d["nomes"])}, referencia=referencia)


def matrizes(pares, titulo=None):
    """pares: [(título, nota real, nota prevista)] → matrizes de confusão lado a lado
    (cor = fração de cada nota real; números = nódulos)."""
    lado, fonte = _tam((4, 3.8), (3.0, 2.9)), _tam(9, 8)
    fig, axes = plt.subplots(1, len(pares), figsize=(lado[0] * len(pares), lado[1]), squeeze=False)
    for ax, (tit, yt, yp) in zip(axes[0], pares):
        M = C.confusion_matrix(yt, yp, labels=C.CLASSES).astype(float)
        V = M / np.maximum(M.sum(1, keepdims=True), 1)
        ax.imshow(V, cmap="Purples", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=fonte,
                        color="white" if V[i, j] > 0.55 else "black")
        ax.set_xticks(range(5)); ax.set_xticklabels(C.CLASSES)
        ax.set_yticks(range(5)); ax.set_yticklabels(C.CLASSES)
        ax.set_xlabel("nota prevista", fontsize=fonte); ax.set_ylabel("nota real", fontsize=fonte)
        ax.set_title(f"{tit}: {len(yt)} nódulos", fontsize=10)
    _fim(fig, titulo)


def matrizes_confusao(nome):
    """Notebooks 1 e 2: matriz de confusão de um modelo gravado em resultados/ (LNDb, 1.ª seed, e holdout)."""
    r = C.carregar_todos().get(nome)
    if r is None:
        print(f"{nome} não está gravado em resultados/ (é um ensemble?): escolham um dos modelos individuais.")
        return
    matrizes([("LNDb (out-of-fold)", C.y.to_numpy()[C.e_lndb], r["oof"].iloc[:, 0].to_numpy()[C.e_lndb]),
              ("Holdout", C.df_holdout["classe"].to_numpy(), r["pred_holdout"])], f"{nome}: matriz de confusão")


def grafico_recall(conjuntos, titulo="Recall de cada nota"):
    """conjuntos: {nome: (nota real, nota prevista)} → barras do recall de cada nota."""
    w = 0.8 / len(conjuntos)
    fig, ax = plt.subplots(figsize=_tam((8, 3), (6.5, 2.4)))
    for j, (nome, (yt, yp)) in enumerate(conjuntos.items()):
        r = C.recall_score(yt, yp, labels=C.CLASSES, average=None, zero_division=0)
        b = ax.bar(np.arange(5) + (j - (len(conjuntos) - 1) / 2) * w, r, w * 0.92, color=CORES[j % len(CORES)],
                   label=nome)
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=2, color="#555555")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"Nota {c}" for c in C.CLASSES])
    ax.set_ylim(0, 1.12); ax.set_ylabel("recall"); ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, ncol=min(len(conjuntos), 3), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _fim(fig, titulo)


def grafico_distribuicao(conjuntos, titulo="Distribuição das notas"):
    """conjuntos: {nome: notas (reais ou previstas)} → % de cada nota em cada conjunto."""
    tab = pd.DataFrame({n: pd.Series(np.asarray(v)).value_counts(normalize=True) for n, v in conjuntos.items()})
    tab = tab.reindex(C.CLASSES).fillna(0)
    cores = [CINZA] + [c for c in CORES if c != CINZA]                 # o real a cinzento, as previsões a roxo
    ax = (tab * 100).plot.bar(figsize=_tam((9.5, 4), (7.5, 2.8)), color=cores[:tab.shape[1]], width=0.82, rot=0)
    ax.set_xticklabels([f"Nota {c}" for c in C.CLASSES])
    ax.set_xlabel(""); ax.set_ylabel("% dos nódulos"); ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, ncol=min(tab.shape[1], 3), loc="upper center",
              bbox_to_anchor=(0.5, _tam(-0.08, -0.1)))
    _fim(None, titulo)


def distribuicao_notas(previstas=None):
    """Distribuição das notas no LIDC e no LNDb e, opcionalmente, de notas previstas {nome: notas}."""
    d = C.df_total
    grafico_distribuicao({"LIDC": d.loc[d["dataset"] == "LIDC", "classe"], "LNDb": d.loc[d["dataset"] == "LNDb", "classe"],
                          **(previstas or {})})


def grafico_importancias(imp, n=15, titulo="Features mais importantes", destacar=None):
    """imp: Series {feature: importância} → as n mais importantes (% do total); 'destacar': prefixo a roxo escuro."""
    s = (imp / imp.sum() * 100).sort_values().tail(n)
    largura, por_barra, base = _tam((7, 0.3, 1.2), (6, 0.22, 0.9))
    fig, ax = plt.subplots(figsize=(largura, por_barra * len(s) + base))
    cores = [ROXO_ESCURO if destacar and str(i).startswith(destacar) else ROXO for i in s.index]
    ax.barh([str(i) for i in s.index], s.to_numpy(), color=cores)
    for k, v in enumerate(s.to_numpy()):
        ax.text(v, k, f" {v:.1f}%", va="center", fontsize=7)
    ax.set_xlabel("% da importância total"); ax.tick_params(axis="y", labelsize=_tam(8, 7)); ax.grid(axis="x", alpha=0.3)
    _fim(fig, titulo)


def ic_bootstrap(y_, p, pacientes, B=1000, seed=42):
    """(valor, limite inferior, limite superior) das 4 métricas, com IC 95% por bootstrap de pacientes."""
    amostras = C.amostras_bootstrap(np.asarray(pacientes), B, seed)
    y_, p = np.asarray(y_), np.asarray(p)
    boot = np.array([C.metr_boot(y_[i], p[i]) for i in amostras])
    return C.metr(y_, p), np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


# ---- Gráficos específicos do notebook 3 (sempre compactos) ----

def grafico_dice(dice, colunas, titulo="Dice com as máscaras dos radiologistas (LNDb), por tamanho do nódulo"):
    """Caixas do Dice de cada método por faixa de tamanho (a mediana a preto)."""
    faixas = [f for f in dice["tamanho"].cat.categories if (dice["tamanho"] == f).any()]
    cores, w = [ROXO_ESCURO, CINZA, ROXO], 0.8 / len(colunas)
    fig, ax = plt.subplots(figsize=(7.5, 3))
    for j, c in enumerate(colunas):
        dados = [dice.loc[dice["tamanho"] == f, c].dropna().to_numpy() for f in faixas]
        cheios = [k for k, d in enumerate(dados) if len(d)]
        bp = ax.boxplot([dados[k] for k in cheios], positions=(np.arange(len(faixas)) + (j - (len(colunas) - 1) / 2) * w)[cheios],
                        widths=w * 0.85, patch_artist=True, showfliers=False, medianprops={"color": "black"})
        for b in bp["boxes"]:
            b.set_facecolor(cores[j % 3])
    ax.set_xticks(range(len(faixas)))
    ax.set_xticklabels([f"{f}\n(n = {int((dice['tamanho'] == f).sum())})" for f in faixas])
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[Patch(color=cores[j % 3], label=c) for j, c in enumerate(colunas)], fontsize=8, frameon=False,
              ncol=len(colunas), loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _fim(fig, titulo)


def grafico_decisoes(paineis, curto=None):
    """Um painel por decisão {título: (tabela de experiencia(), escolhas[, ordem das opções])}: S ± desvio de cada
    modelo em cada opção; a opção escolhida tem contorno preto."""
    curto = curto or (lambda o: o)
    fig, axes = plt.subplots(1, len(paineis), figsize=(4.3 * len(paineis), 3.3), sharey=True, squeeze=False)
    for ax, (titulo, (t, escolhas, *ordem)) in zip(axes[0], paineis.items()):
        pares = [str(i).split(" | ", 1) for i in t.index]
        modelos, opcoes = list(dict.fromkeys(a for a, _ in pares)), list(dict.fromkeys(b for _, b in pares))
        ordem = ordem[0] if ordem else []
        opcoes = [o for o in ordem if o in opcoes] + [o for o in opcoes if o not in ordem]
        w = 0.8 / len(opcoes)
        for j, o in enumerate(opcoes):
            cor = CORES[j % len(CORES)]
            for k, m in enumerate(modelos):
                if f"{m} | {o}" in t.index:
                    esc = escolhas == o if isinstance(escolhas, str) else (escolhas or {}).get(m) == o
                    ax.errorbar(k + (j - (len(opcoes) - 1) / 2) * w, t.loc[f"{m} | {o}", "S"],
                                yerr=t.loc[f"{m} | {o}", "S_desvio"], fmt="o", color=cor, ms=7 if esc else 4,
                                mec="black" if esc else cor, mew=1.3 if esc else 0, capsize=2, lw=1)
            ax.plot([], [], "o", color=cor, label=curto(o))
        ax.plot([], [], "o", mfc="none", mec="black", mew=1.3, ms=7, label="escolhida")
        ax.set_xticks(range(len(modelos))); ax.set_xticklabels(modelos, rotation=30, ha="right")
        ax.set_title(titulo, fontsize=9); ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    axes[0][0].set_ylabel("S no LNDb (média ± desvio)", fontsize=8)
    _fim(fig)


def grafico_limiares(paineis, limiares, titulo="Previsões contínuas por nota real e limiares"):
    """paineis: [(título, previsões contínuas, nota real)]. Caixas e pontos por nota real; limiares a tracejado."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(paineis), figsize=(4.4 * len(paineis), 3.2), sharey=True, squeeze=False)
    for ax, (tit, s, y_) in zip(axes[0], paineis):
        s, y_ = np.asarray(s, float), np.asarray(y_)
        pres = [c for c in C.CLASSES if (y_ == c).any()]
        ax.boxplot([s[y_ == c] for c in pres], positions=pres, widths=0.5, showfliers=False,
                   medianprops={"color": "black"}, boxprops={"color": "#7f7f7f"})
        for c in pres:
            ax.scatter(c + rng.uniform(-0.17, 0.17, (y_ == c).sum()), s[y_ == c], s=9, color=ROXO_ESCURO, alpha=0.5, zorder=3)
        for k, t in enumerate(limiares):
            ax.axhline(t, color=ROXO_ESCURO, ls="--", lw=0.8)
            ax.text(5.45, t, f"{k + 1}–{k + 2}", va="center", fontsize=7, color=ROXO_ESCURO)
        ax.set_xticks(C.CLASSES); ax.set_xlim(0.5, 5.7); ax.set_xlabel("nota real", fontsize=8)
        ax.set_title(f"{tit} ({len(s)} nódulos)", fontsize=9); ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("previsão contínua", fontsize=8)
    _fim(fig, titulo + " (a tracejado: limiares entre notas)")


def grafico_resumo(blocos, teto=None, titulo="Desempenho do modelo final"):
    """Notebook 3 (13.6): várias estimativas do MESMO modelo {nome: (valor, inferior, superior)}, uma cor por
    estimativa (sem marcar a "melhor", porque não há modelos a comparar), com o valor escrito por cima do ponto e o
    teto dos radiologistas a ponteado ({métrica: valor})."""
    nomes = list(blocos)
    cores = [ROXO_ESCURO, ROXO, "#7f7f7f", "#3f3f3f"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 0.45 * len(nomes) + 1.5), sharey=True)
    ypos = np.arange(len(nomes))[::-1]
    for j, (ax, m) in enumerate(zip(axes, M4)):
        for i, n in enumerate(nomes):
            v, lo, hi = blocos[n]
            c = cores[i % len(cores)]
            ax.plot([lo[j], hi[j]], [ypos[i]] * 2, color=c, lw=3, alpha=0.85, solid_capstyle="round")
            ax.scatter(v[j], ypos[i], color=c, s=36, zorder=3, edgecolor="white", linewidth=0.8)
            ax.annotate(f"{v[j]:.2f}", (v[j], ypos[i]), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color="#333333",
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
        if teto is not None:
            ax.axvline(teto[m], color="black", ls=":", lw=1.1)
        ax.set_ylim(-0.6, len(nomes) - 0.25) # as linhas ficam próximas, sem espaço vazio
        ax.margins(x=0.08)
        ax.set_title(NOMES[m] + (" (menor é melhor)" if m == "mae" else ""), fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(nomes, fontsize=8)
    leg = [plt.Line2D([], [], color=cores[i % len(cores)], lw=3, marker="o", label=n) for i, n in enumerate(nomes)]
    if teto is not None:
        leg.append(plt.Line2D([], [], color="black", ls=":", label="teto (radiologista vs. colegas)"))
    fig.legend(handles=leg, loc="lower center", ncol=len(leg), fontsize=8, frameon=False)
    _fim(fig, titulo, rect=(0, 0.12, 1, 1))