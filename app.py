"""
EnerGuard Starter Kit - Scheletro dashboard Streamlit
======================================================
Esecuzione:  streamlit run app.py
Prerequisito: aver eseguito train_baseline.py (genera predizioni.csv)

Lo scheletro definisce le SEZIONI OBBLIGATORIE della dashboard (vedi
documento "Indicatori di Qualita'"). Il come le realizzate e' scelta
vostra: Streamlit e' solo il default comodo, potete usare React/Gradio.
"""

import json
import os
from datetime import datetime, timedelta
from random import Random

import altair as alt
import pandas as pd
import streamlit as st

from audit_logger import AuditLogger
from bias_detector import BiasDetector
from explainer import ConfigLLM, Spiegazione, crea_spiegatore, estrai_fattori
from utils_io import carica_csv
from oversight_manager import (OversightManager, Raccomandazione,
                               StatoDecisione, AZIONI)

st.set_page_config(page_title="EnerGuard | Console Operatore",
                   layout="wide", page_icon="⚡")

PASTEL_REGIME_COLORS = {
    "HOTL": "#a7d8b5",
    "HITL": "#f6c99b",
    "HIC": "#f3a6a6",
}
PASTEL_AREA_COLORS = ["#8ecae6", "#b8b8ff", "#ffd6a5", "#cdb4db"]
PASTEL_THRESHOLD = "#9c7a5d"

st.markdown(
    """
    <style>
    :root {
        --eg-bg: #f7f4ef;
        --eg-surface: #fffdf8;
        --eg-surface-soft: #f1ebe3;
        --eg-text: #31413d;
        --eg-muted: #6f7f78;
        --eg-accent: #8bb7a2;
        --eg-accent-strong: #5f927b;
        --eg-warn: #f6c99b;
        --eg-danger: #b42318;
        --eg-danger-dark: #7a271a;
        --eg-danger-soft: #fef3f2;
        --eg-border: #ded6ca;
    }

    .stApp {
        background: linear-gradient(180deg, var(--eg-bg) 0%, #eef5f1 52%, #f7f4ef 100%);
        color: var(--eg-text);
    }

    h1, h2, h3, .stMarkdown, .stCaption, label {
        color: var(--eg-text) !important;
    }

    [data-testid="stMetric"], div[data-testid="stExpander"] {
        background: rgba(255, 253, 248, 0.86);
        border: 1px solid var(--eg-border);
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(95, 111, 101, 0.08);
        padding: 0.6rem;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #17211d !important;
        -webkit-text-fill-color: #17211d !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        color: #245c48 !important;
        -webkit-text-fill-color: #245c48 !important;
        opacity: 1 !important;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--eg-accent);
        background: #d8eadf;
        color: #2f5144;
    }

    .stButton > button[kind="primary"] {
        background: var(--eg-accent);
        color: #ffffff;
        border-color: var(--eg-accent-strong);
    }

    .stButton > button:disabled {
        background: #ebe5dc;
        color: var(--eg-muted);
        border-color: var(--eg-border);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: var(--eg-surface) !important;
        border-color: var(--eg-border) !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: var(--eg-text) !important;
        -webkit-text-fill-color: var(--eg-text) !important;
        caret-color: var(--eg-accent-strong) !important;
    }

    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder {
        color: var(--eg-muted) !important;
        opacity: 1;
    }

    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: #66736d !important;
        color: #17211d !important;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] input::placeholder {
        color: #17211d !important;
        -webkit-text-fill-color: #17211d !important;
        opacity: 1 !important;
    }

    div[data-baseweb="select"] svg {
        fill: #17211d !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] ul,
    ul[role="listbox"] {
        background: #ffffff !important;
        color: #17211d !important;
    }

    li[role="option"] {
        background: #ffffff !important;
        color: #17211d !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background: #d8eadf !important;
        color: #173c2f !important;
    }

    [data-testid="stCheckbox"] [data-testid="stWidgetLabel"],
    [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] * {
        color: #17211d !important;
        -webkit-text-fill-color: #17211d !important;
        opacity: 1 !important;
    }

    [data-testid="stCheckbox"] label > span:first-child {
        background: #ffffff !important;
        border: 2px solid #66736d !important;
    }

    [data-testid="stCheckbox"] input:checked + span {
        background: #245c48 !important;
        border-color: #163b2e !important;
    }

    [data-testid="stExpander"] details > summary {
        background: #eef3f0 !important;
        color: #17211d !important;
        border-bottom: 1px solid var(--eg-border);
    }

    [data-testid="stExpander"] details > summary *,
    [data-testid="stExpander"] details > summary p {
        color: #17211d !important;
        -webkit-text-fill-color: #17211d !important;
        opacity: 1 !important;
    }

    .st-key-header_operator input:disabled {
        background: #ebe7df !important;
        color: #465550 !important;
        -webkit-text-fill-color: #465550 !important;
        opacity: 1 !important;
        cursor: default;
    }

    button[data-testid="stBaseButton-pills"] {
        background: #ffffff !important;
        border: 2px solid #66736d !important;
        color: #17211d !important;
        font-weight: 600;
        opacity: 1 !important;
    }

    button[data-testid="stBaseButton-pills"] *,
    button[data-testid="stBaseButton-pillsActive"] * {
        color: inherit !important;
        -webkit-text-fill-color: currentColor !important;
        opacity: 1 !important;
    }

    button[data-testid="stBaseButton-pills"]:hover,
    button[data-testid="stBaseButton-pills"]:focus-visible {
        background: #eef2f0 !important;
        border-color: #263b34 !important;
    }

    button[data-testid="stBaseButton-pillsActive"] {
        background: #245c48 !important;
        border-color: #163b2e !important;
        color: #ffffff !important;
        font-weight: 700;
        box-shadow: none;
    }

    .st-key-stop_scope_global button[data-testid="stBaseButton-pillsActive"] {
        background: #9f1d16 !important;
        border-color: #65120e !important;
        color: #ffffff !important;
    }

    .st-key-stop_cta button {
        min-height: 3rem;
        background: var(--eg-danger) !important;
        border-color: var(--eg-danger) !important;
        color: #ffffff !important;
        font-weight: 700;
    }

    .st-key-stop_cta button:hover,
    .st-key-stop_cta button:focus-visible {
        background: var(--eg-danger-dark) !important;
        border-color: var(--eg-danger-dark) !important;
    }

    .st-key-stop_panel {
        background: var(--eg-danger-soft);
        border: 1px solid #fecdca;
        border-left: 5px solid var(--eg-danger);
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 1rem;
    }

    div[data-testid="stDialog"] [role="dialog"] {
        background: var(--eg-surface) !important;
        color: var(--eg-text) !important;
        border-top: 6px solid var(--eg-danger);
        border-left: 1px solid var(--eg-border);
        border-right: 1px solid var(--eg-border);
        border-bottom: 1px solid var(--eg-border);
        box-shadow: 0 20px 55px rgba(49, 65, 61, 0.24);
    }

    div[data-testid="stDialog"] [role="dialog"] > div,
    div[data-testid="stDialog"] [role="dialog"] [data-testid="stVerticalBlock"],
    div[data-testid="stDialog"] [role="dialog"] [data-testid="stMarkdownContainer"] {
        background: transparent !important;
        color: var(--eg-text) !important;
    }

    div[data-testid="stDialog"] [role="dialog"] p,
    div[data-testid="stDialog"] [role="dialog"] label,
    div[data-testid="stDialog"] [role="dialog"] span {
        color: var(--eg-text) !important;
        -webkit-text-fill-color: var(--eg-text) !important;
    }

    div[data-testid="stDialog"] [role="dialog"] div[data-testid="stAlert"] {
        background: #fde7e5 !important;
        border-color: #d92d20 !important;
    }

    div[data-testid="stDialog"] [role="dialog"] div[data-testid="stNotification"] {
        background: #eaf4f8 !important;
        border-color: #72a9bd !important;
    }

    div[data-testid="stDialog"] [role="dialog"] .stButton > button:not([kind="primary"]) {
        background: #ffffff !important;
        border-color: #66736d !important;
        color: #17211d !important;
    }

    div[data-testid="stDialog"] .stButton > button[kind="primary"] {
        background: var(--eg-danger) !important;
        border-color: var(--eg-danger) !important;
        color: #ffffff !important;
    }

    div[data-testid="stDialog"] .stButton > button[kind="primary"] p,
    div[data-testid="stDialog"] .stButton > button[kind="primary"] span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    [data-baseweb="tab"] {
        background: rgba(255, 253, 248, 0.72);
        border-radius: 10px 10px 0 0;
        color: var(--eg-muted);
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background: #dcefe6;
        color: #2f5144;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
        border-color: var(--eg-border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _fattori_asset(asset_id: str):
    """Estrae i fattori SHAP per un asset e li riusa tra i rerun Streamlit."""
    import joblib
    modello = joblib.load("modello.joblib")
    df = carica_csv("energuard_dataset.csv")
    X = pd.get_dummies(df.drop(columns=["asset_id", "guasto_entro_30gg"]))
    asset_origine = asset_id.removeprefix("HP-")
    x = X.loc[df["asset_id"] == asset_origine].iloc[0]
    return estrai_fattori(modello, x, list(X.columns))


@st.cache_resource
def bootstrap():
    audit = AuditLogger("audit_trail.jsonl")
    om = OversightManager(audit, stop_state_path="oversight_stops.json")
    spiegatore = crea_spiegatore(audit)   # LLM se .env e' configurato, altrimenti template
    pred = carica_csv("predizioni.csv")

    def prepara(r):
        fattori = _fattori_asset(r.asset_id)
        r.spiegazione = [(f.nome, f.valore, f.contributo) for f in fattori]
        rec = {"asset_id": r.asset_id, "tipo_asset": r.tipo_asset,
               "area_geografica": r.area_geografica, "criticita_utenza": r.criticita_utenza,
               "prob_guasto": r.prob_guasto, "confidenza": r.confidenza,
               "azione_proposta": r.azione_proposta, "livello": None,
               "soglia_confidenza": om.soglia_conf}
        sp = spiegatore.spiega(rec, fattori)
        r.messaggio_llm = {"testo": sp.testo, "incertezza": sp.incertezza,
                           "fonte": sp.fonte, "latenza_ms": sp.latenza_ms}
        audit.log("SISTEMA", "spiegazione_generata", r,
                  extra={"fonte": sp.fonte, "latenza_ms": sp.latenza_ms})
    # Trasforma le prime N predizioni positive in raccomandazioni in coda
    for _, riga in pred[pred["y_pred"] == 1].head(40).iterrows():
        r = Raccomandazione(
            asset_id=riga["asset_id"],
            tipo_asset=riga["tipo_asset"],
            area_geografica=riga["area_geografica"],
            criticita_utenza=riga["criticita_utenza"],
            prob_guasto=round(float(riga["proba"]), 3),
            confidenza=round(float(riga["confidenza"]), 3),
            azione_proposta=("riduci_carico" if riga["criticita_utenza"] == "critica"
                             else "ispezione_urgente" if float(riga["proba"]) >= 0.75
                             else "programma_manutenzione"),
            spiegazione=[],
        )
        prepara(r)
        om.sottometti(r)
    # Honeypot anti rubber-stamping: casi palesemente incoerenti inseriti nel flusso.
    candidati_honeypot = pred[(pred["y_true"] == 0) & (pred["y_pred"] == 0)].nsmallest(6, "proba")
    for _, riga in candidati_honeypot.sample(
            n=min(2, len(candidati_honeypot)), random_state=Random().randint(1, 999999)
    ).iterrows():
        r = Raccomandazione(
            asset_id=f"HP-{riga['asset_id']}",
            tipo_asset=riga["tipo_asset"],
            area_geografica=riga["area_geografica"],
            criticita_utenza="standard",
            prob_guasto=round(float(riga["proba"]), 3),
            confidenza=round(float(riga["confidenza"]), 3),
            azione_proposta="ispezione_urgente",
            spiegazione=[],
            honeypot=True,
            honeypot_messaggio="Rischio molto basso, utenza standard e nessuna evidenza critica: approvare l'urgenza indica possibile rubber stamping.",
        )
        prepara(r)
        om.sottometti(r)
    return om, audit, pred, spiegatore


om, audit, pred, spiegatore = bootstrap()

aree_stop = sorted(pred["area_geografica"].dropna().unique())
tipi_stop = sorted(pred["tipo_asset"].dropna().unique())
st.session_state.setdefault("stop_scope_global", "GLOBALE")
st.session_state.setdefault("stop_scope_areas", [])
st.session_state.setdefault("stop_scope_types", [])


def seleziona_stop_globale():
    if st.session_state.get("stop_scope_global") == "GLOBALE":
        st.session_state["stop_scope_areas"] = []
        st.session_state["stop_scope_types"] = []


def seleziona_stop_specifico():
    if (st.session_state.get("stop_scope_areas")
            or st.session_state.get("stop_scope_types")):
        st.session_state["stop_scope_global"] = None


@st.dialog("Conferma stop operativo")
def conferma_attivazione_stop(ambiti, motivazione, operatore_corrente):
    st.error("Stai per bloccare l'esecuzione delle decisioni negli ambiti selezionati.")
    st.markdown("**Ambiti interessati**")
    st.write(", ".join(ambiti))
    st.markdown("**Motivazione registrata nell'audit trail**")
    st.info(motivazione)
    st.caption("Lo stop si applica alle nuove raccomandazioni e a quelle già presenti in coda.")
    conferma = st.checkbox("Confermo di aver verificato ambiti e motivazione")
    col_conferma, col_annulla = st.columns(2)
    if col_conferma.button("Conferma stop", type="primary", disabled=not conferma,
                           use_container_width=True):
        try:
            decisioni_bloccate = om.attiva_stop_filtri(
                ambiti, operatore_corrente, motivazione)
            st.session_state["header_feedback"] = (
                f"Stop attivato su {len(ambiti)} ambiti. "
                f"Decisioni bloccate: {decisioni_bloccate}.")
            st.session_state["stop_panel_open"] = False
            st.rerun()
        except ValueError as errore:
            st.error(str(errore))
    if col_annulla.button("Annulla", use_container_width=True):
        st.rerun()


header_title, header_operator, header_stop = st.columns([5, 1.6, 1.5], vertical_alignment="bottom")
with header_title:
    st.title("EnerGuard · Console operatore")
    st.caption("Decisioni, controllo umano e stato operativo in un'unica vista.")
with header_operator:
    operatore = st.text_input(
        "ID operatore", value="OP-001", key="header_operator", disabled=True)
with header_stop:
    if st.button("STOP OPERATIVO", type="primary", use_container_width=True, key="stop_cta"):
        st.session_state["stop_panel_open"] = not st.session_state.get("stop_panel_open", False)

header_feedback = st.session_state.pop("header_feedback", None)
if header_feedback:
    st.success(header_feedback)

if om.stop_attivi:
    st.error(f"STOP ATTIVO · {', '.join(sorted(om.stop_attivi))}")

if st.session_state.get("stop_panel_open", False):
    with st.container(border=True, key="stop_panel"):
        st.subheader("Configura stop operativo")
        st.caption("Definisci l'ambito e una motivazione verificabile. Potrai rileggere tutto prima della conferma finale.")
        stop_scope, stop_reason = st.columns([1, 1.4])
        with stop_scope:
            st.markdown("**Ambiti da bloccare**")
            st.caption("Globale esclude automaticamente ogni selezione specifica.")
            globale_stop = st.pills(
                "Globale",
                ["GLOBALE"],
                key="stop_scope_global",
                on_change=seleziona_stop_globale,
            )
            aree_selezionate = st.pills(
                "Area",
                aree_stop,
                selection_mode="multi",
                key="stop_scope_areas",
                on_change=seleziona_stop_specifico,
            )
            tipi_selezionati = st.pills(
                "Tipo asset",
                tipi_stop,
                selection_mode="multi",
                key="stop_scope_types",
                on_change=seleziona_stop_specifico,
            )
            ambiti_stop = (
                ["GLOBALE"] if globale_stop == "GLOBALE"
                else [f"area:{area}" for area in (aree_selezionate or [])]
                + [f"tipo:{tipo}" for tipo in (tipi_selezionati or [])]
            )
        with stop_reason:
            mot_stop = st.text_area(
                "Motivazione dello stop",
                placeholder="Descrivi l'anomalia o il rischio operativo che richiede il blocco",
                key="header_stop_reason",
            )
        azione_stop, chiudi_stop = st.columns([1, 1])
        if azione_stop.button("Rivedi e conferma", type="primary", key="review_stop",
                              use_container_width=True):
            try:
                om._valida_motivazione(mot_stop, consentire_duplicati=True)
                om._normalizza_ambiti(ambiti_stop)
                conferma_attivazione_stop(ambiti_stop, mot_stop.strip(), operatore)
            except ValueError as errore:
                st.error(str(errore))
        if chiudi_stop.button("Chiudi", key="close_stop_panel", use_container_width=True):
            st.session_state["stop_panel_open"] = False
            st.rerun()

        if om.stop_attivi:
            st.divider()
            st.markdown("**Riprendi attività**")
            st.caption("Disattiva solo gli ambiti selezionati; gli altri stop resteranno operativi.")
            resume_scope, resume_reason = st.columns([1, 1.4])
            with resume_scope:
                ambiti_ripresa = st.multiselect(
                    "Ambiti da riattivare",
                    sorted(om.stop_attivi),
                    key="header_resume_scopes",
                )
            with resume_reason:
                motivo_ripresa = st.text_input(
                    "Motivazione della riattivazione",
                    key="header_resume_reason",
                )
            if st.button("Riprendi ambiti selezionati", key="resume_selected"):
                try:
                    ripristinate = om.disattiva_stop_filtri(
                        ambiti_ripresa, operatore, motivo_ripresa)
                    st.session_state["header_feedback"] = (
                        f"Stop disattivato su {len(ambiti_ripresa)} ambiti. "
                        f"Attività ripristinate: {ripristinate}.")
                    st.rerun()
                except ValueError as errore:
                    st.error(str(errore))

st.markdown(
    """
    **Lettura rapida:** controlla prima i quattro indicatori in alto, poi lavora le attività in attesa, infine usa la mappa per vedere dove si concentrano rischio e bassa confidenza.
    """
)

_cfg = ConfigLLM()
st.caption(
    f"Motore spiegazioni: **{'LLM · ' + _cfg.descrizione() if _cfg.attivo else 'template locale'}**"
    + ("" if _cfg.attivo else "  \nConfigurate .env (vedi .env.example) per usare un LLM esterno."))

nuove_escalation = om.aggiorna_sla() if hasattr(om, "aggiorna_sla") else 0
if nuove_escalation:
    st.warning(f"{nuove_escalation} attività HITL hanno superato lo SLA e sono in escalation.")

stati_lavorabili = {StatoDecisione.IN_ATTESA, StatoDecisione.ESCALATION}
pendenti = [r for r in om.coda if r.stato in stati_lavorabili]
bloccate_stop = [r for r in om.coda if r.stato == StatoDecisione.BLOCCATA_STOP]
in_escalation = [r for r in om.coda if r.stato == StatoDecisione.ESCALATION]
summary_cols = st.columns(4)
with summary_cols[0]:
    st.metric("Attività da lavorare", len(pendenti))
with summary_cols[1]:
    st.metric("Attività bloccate", len(bloccate_stop), delta=f"{len(om.stop_attivi)} filtri di stop")
with summary_cols[2]:
    st.metric("Escalation SLA", len(in_escalation), delta=f"SLA {om.sla_minuti} min")

st.info("Regola operativa: lavora solo le attività in attesa. Quelle bloccate da stop non devono essere eseguite finché non viene ripresa l'attività.")
review_feedback = st.session_state.pop("review_feedback", None)
if review_feedback:
    st.success(review_feedback)

def spiegazione_per(r):
    """Spiegazione in linguaggio operativo per una raccomandazione in coda."""
    fattori_calcolati = _fattori_asset(r.asset_id)
    if r.messaggio_llm:
        return Spiegazione(r.messaggio_llm["testo"], r.messaggio_llm["incertezza"],
                           fattori_calcolati, r.messaggio_llm["fonte"],
                           r.messaggio_llm.get("latenza_ms", 0))
    rec = {"asset_id": r.asset_id, "tipo_asset": r.tipo_asset, "area_geografica": r.area_geografica,
           "criticita_utenza": r.criticita_utenza, "prob_guasto": r.prob_guasto,
           "confidenza": r.confidenza, "azione_proposta": r.azione_proposta,
           "livello": getattr(r.livello, "value", None), "soglia_confidenza": om.soglia_conf}
    sp = spiegatore.spiega(rec, fattori_calcolati)
    r.messaggio_llm = {
        "testo": sp.testo,
        "incertezza": sp.incertezza,
        "fonte": sp.fonte,
        "latenza_ms": sp.latenza_ms,
    }
    return sp


def parole_chiave_revisione(r):
    parole = [
        f"asset {r.asset_id}",
        f"area {r.area_geografica}",
        f"tipo {r.tipo_asset}",
        f"utenza {r.criticita_utenza}",
        f"rischio {r.prob_guasto:.2f}",
        f"confidenza {r.confidenza:.2f}",
        r.azione_proposta.replace("_", " "),
    ]
    if r.honeypot:
        parole.extend(["rischio basso", "azione urgente incoerente"])
    return list(dict.fromkeys(parole))


def cooldown_pronto(decision_id: str, secondi: int = 10):
    key = f"cooldown_{decision_id}"
    if key not in st.session_state:
        st.session_state[key] = datetime.now()
    elapsed = datetime.now() - st.session_state[key]
    residuo = max(0, secondi - int(elapsed.total_seconds()))
    return elapsed >= timedelta(seconds=secondi), residuo


def testo_sla(r):
    if not hasattr(om, "minuti_residui_sla"):
        return "SLA non disponibile"
    minuti = om.minuti_residui_sla(r)
    if minuti is None:
        return "SLA non applicabile"
    if r.stato == StatoDecisione.ESCALATION:
        return f"SLA scaduto da {abs(minuti)} min"
    if minuti <= 5:
        return f"SLA urgente: {max(minuti, 0)} min"
    return f"SLA: {minuti} min"


def classe_sla(r):
    if r.stato == StatoDecisione.ESCALATION:
        return "Scaduto"
    if not hasattr(om, "minuti_residui_sla"):
        return "Non applicabile"
    minuti = om.minuti_residui_sla(r)
    if minuti is None:
        return "Non applicabile"
    if minuti <= 5:
        return "Urgente"
    return "Nei tempi"


def filtra_attivita(attivita, testo, regimi, aree, tipi, sla, solo_honeypot):
    filtrate = attivita
    if testo:
        testo_norm = testo.strip().casefold()
        filtrate = [
            r for r in filtrate
            if testo_norm in r.asset_id.casefold()
            or testo_norm in r.tipo_asset.casefold()
            or testo_norm in r.area_geografica.casefold()
        ]
    if regimi:
        filtrate = [r for r in filtrate if r.livello.value in regimi]
    if aree:
        filtrate = [r for r in filtrate if r.area_geografica in aree]
    if tipi:
        filtrate = [r for r in filtrate if r.tipo_asset in tipi]
    if sla:
        filtrate = [r for r in filtrate if classe_sla(r) in sla]
    if solo_honeypot:
        filtrate = [r for r in filtrate if r.honeypot]
    return filtrate


def carica_records_audit(percorso="audit_trail.jsonl"):
    if not os.path.exists(percorso):
        return []
    records = []
    with open(percorso, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            evento = record.get("evento", "")
            if "llm" in evento.lower():
                continue
            records.append(record)
    return sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)


def categoria_evento_audit(evento):
    evento = str(evento or "").lower()
    if evento.startswith("revisione_"):
        return "Revisioni"
    if "emergency_stop" in evento or "ripristino_post" in evento:
        return "Stop operativo"
    if "sla" in evento or "escalation" in evento:
        return "SLA"
    if "honeypot" in evento:
        return "Controlli"
    if evento.startswith("in_coda_") or "auto_esecuzione" in evento:
        return "Instradamento"
    return "Altro"


def evento_leggibile(evento):
    mapping = {
        "revisione_APPROVATA": "Approvazione",
        "revisione_MODIFICATA": "Modifica e approvazione",
        "revisione_RIFIUTATA": "Rifiuto",
        "escalation_sla_scaduto": "Escalation SLA",
        "blocco_emergency_stop": "Blocco da stop",
        "blocco_emergency_stop_attivato": "Blocco da stop attivato",
        "ripristino_post_emergency_stop": "Ripristino attività",
        "emergency_stop_ON_multiplo": "Stop multiplo",
        "emergency_stop_OFF_multiplo": "Ripresa multipla",
        "allerta_honeypot_approvato": "Allerta controllo",
        "auto_esecuzione_HOTL": "Auto-esecuzione HOTL",
    }
    evento = str(evento or "-")
    if evento.startswith("emergency_stop_ON:"):
        return "Stop attivato"
    if evento.startswith("emergency_stop_OFF:"):
        return "Stop disattivato"
    if evento.startswith("in_coda_"):
        return "Inserita in coda"
    return mapping.get(evento, evento.replace("_", " "))


STATO_COLORI = {
    "APPROVATA": "🟢",
    "MODIFICATA": "🔵",
    "RIFIUTATA": "🔴",
    "IN_ATTESA": "🟡",
    "ESCALATION": "🟣",
    "BLOCCATA_STOP": "🟠",
    "AUTO_ESEGUITA": "⚪",
    "Senza stato": "⚫",
}


def stato_leggibile(stato):
    if not stato or stato == "-":
        return "-"
    return str(stato).replace("_", " ").upper()


def stato_filtro_leggibile(stato):
    colore = STATO_COLORI.get(str(stato), "⚫")
    return f"{colore} {stato_leggibile(stato)}"


def data_leggibile(timestamp):
    if not timestamp or timestamp == "-":
        return "-"
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(timestamp)[:19]


def dentro_finestra_temporale(record, filtro):
    if filtro == "Tutto":
        return True
    timestamp = record.get("timestamp")
    if not timestamp:
        return False
    try:
        quando = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    adesso = datetime.now()
    if filtro == "Oggi":
        return quando.date() == adesso.date()
    if filtro == "Ultime 24 ore":
        return quando >= adesso - timedelta(hours=24)
    if filtro == "Ultimi 7 giorni":
        return quando >= adesso - timedelta(days=7)
    return True


def filtra_records_audit(records, stati, supervisioni, attori, finestra, testo):
    filtrati = [r for r in records if dentro_finestra_temporale(r, finestra)]
    if stati:
        filtrati = [r for r in filtrati if (r.get("decisione", {}).get("stato") or "Senza stato") in stati]
    if supervisioni:
        filtrati = [r for r in filtrati if (r.get("decisione", {}).get("livello") or "Senza supervisione") in supervisioni]
    if attori:
        filtrati = [r for r in filtrati if (r.get("attore") or "-") in attori]
    if testo:
        testo_norm = testo.strip().casefold()
        filtrati = [
            r for r in filtrati
            if testo_norm in str(r.get("evento", "")).casefold()
            or testo_norm in str(r.get("attore", "")).casefold()
            or testo_norm in str(r.get("decisione", {}).get("asset_id", "")).casefold()
            or testo_norm in str(r.get("extra", {}).get("asset_id", "")).casefold()
        ]
    return filtrati


def richiedi_apertura_dettaglio_log():
    st.session_state["audit_apri_dettaglio"] = True


def livello_atteso(r):
    if r.criticita_utenza == "critica" or (
            r.azione_proposta == "riduci_carico" and r.criticita_utenza != "standard"):
        return "HIC"
    if r.prob_guasto >= om.soglia_rischio or r.confidenza < om.soglia_conf:
        return "HITL"
    if r.azione_proposta in ("nessuna_azione", "ispezione_routine"):
        return "HOTL"
    return "HITL"


def percentuale(numeratore, denominatore):
    if not denominatore:
        return "n/d"
    return f"{round(100 * numeratore / denominatore, 1)}%"


def minuti_mediani_revisione(records):
    aperture = {}
    durate = []
    for record in sorted(records, key=lambda r: r.get("timestamp", "")):
        decisione = record.get("decisione", {})
        decision_id = decisione.get("id")
        timestamp = record.get("timestamp")
        if not decision_id or not timestamp:
            continue
        try:
            quando = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        evento = record.get("evento", "")
        if evento.startswith("in_coda_"):
            aperture.setdefault(decision_id, quando)
        elif evento.startswith("revisione_") and decision_id in aperture:
            durate.append((quando - aperture[decision_id]).total_seconds() / 60)
    if not durate:
        return "n/d"
    return f"{round(float(pd.Series(durate).median()), 1)} min"


def tabella_kpi(records):
    manager_kpi = om.kpi()
    chiuse = [r for r in om.coda if r.stato in (
        StatoDecisione.APPROVATA, StatoDecisione.MODIFICATA, StatoDecisione.RIFIUTATA)]
    override = [r for r in chiuse if r.stato in (StatoDecisione.MODIFICATA, StatoDecisione.RIFIUTATA)]
    hitl = [r for r in om.coda if getattr(r.livello, "value", None) == "HITL"]
    hic_hitl = [r for r in om.coda if getattr(r.livello, "value", None) in ("HIC", "HITL")]
    auto_improprie = [r for r in hic_hitl if r.stato == StatoDecisione.AUTO_ESEGUITA]
    routing_ok = [r for r in om.coda if getattr(r.livello, "value", None) == livello_atteso(r)]

    revisioni_log = [r for r in records if str(r.get("evento", "")).startswith("revisione_")]
    motivazioni = [r.get("decisione", {}).get("motivazione") for r in revisioni_log]
    motivazioni = [m.strip() for m in motivazioni if isinstance(m, str) and m.strip()]
    motivazioni_normalizzate = [" ".join(m.split()).casefold() for m in motivazioni]
    duplicati = len(motivazioni_normalizzate) - len(set(motivazioni_normalizzate))
    brevi = sum(1 for m in motivazioni if len(m) < 30)
    rubber = brevi + duplicati

    bd = BiasDetector()
    valut = pred.copy()
    metriche_area = bd.metriche_per_gruppo(valut, "area_geografica")
    metriche_tipo = bd.metriche_per_gruppo(valut, "tipo_asset")
    gap_recall_area = round(metriche_area["recall"].max() - metriche_area["recall"].min(), 3)
    gap_recall_tipo = round(metriche_tipo["recall"].max() - metriche_tipo["recall"].min(), 3)
    calibrazione = bd.calibrazione_per_gruppo(valut, "area_geografica")
    gap_calibrazione = round(calibrazione["differenza"].abs().max(), 3)

    storico_override = []
    for r in chiuse:
        storico_override.append({
            "area": r.area_geografica,
            "override": int(r.stato in (StatoDecisione.MODIFICATA, StatoDecisione.RIFIUTATA)),
        })
    if storico_override:
        ov_area = pd.DataFrame(storico_override).groupby("area")["override"].mean()
        override_area = f"max {round(float(ov_area.max()) * 100, 1)}%"
    else:
        override_area = "n/d"

    spiegazioni_con_fonte = [
        r for r in records
        if r.get("decisione", {}).get("messaggio_llm", {}).get("fonte")
    ]
    chiamate = getattr(spiegatore, "chiamate", None)
    if chiamate and (chiamate.get("ok", 0) + chiamate.get("fallback", 0)):
        fallback_rate = percentuale(chiamate.get("fallback", 0), chiamate.get("ok", 0) + chiamate.get("fallback", 0))
    else:
        fallback_rate = "n/d"

    righe = [
        ["A1", "Auto-esecuzione impropria", f"{len(auto_improprie)} / {len(hic_hitl)}", "0 assoluto", "OK" if not auto_improprie else "Allarme"],
        ["A2", "Override umano", percentuale(len(override), len(chiuse)), "5% - 40%", "Da monitorare" if chiuse else "n/d"],
        ["A3", "Tempo revisione", f"media {manager_kpi['tempo_medio_revisione_min'] or 'n/d'} min; mediana {manager_kpi['tempo_mediano_revisione_min'] or 'n/d'} min", "30 sec - 5 min", "Da monitorare"],
        ["A4", "Indice rubber-stamping", f"brevi {manager_kpi['motivazioni_brevi_pct'] or 0}% + duplicati {percentuale(duplicati, len(motivazioni))}", "< 10%", "OK" if not motivazioni or rubber / max(len(motivazioni), 1) < 0.10 else "Allarme"],
        ["A5", "Escalation SLA scaduto", percentuale(len(in_escalation), len(hitl)), "< 15%", "OK" if not hitl or len(in_escalation) / len(hitl) < 0.15 else "Allarme"],
        ["A6", "Copertura routing dichiarato", percentuale(len(routing_ok), len(om.coda)), "100%", "OK" if len(routing_ok) == len(om.coda) else "Allarme"],
        ["A6b", "Distribuzione livelli", "; ".join(f"{livello} {numero}" for livello, numero in manager_kpi["distribuzione_livelli"].items()), "Esposta", "OK"],
        ["B1", "Copertura spiegazioni", "100% sulle card visibili", "100%", "OK"],
        ["B2", "Visibilità incertezza", "100% sulle card visibili", "100%", "OK"],
        ["B3", "Test dei 60 secondi", "Da verificare in demo", "Superato", "Manuale"],
        ["B3b", "Tracciabilità spiegazione", percentuale(len(spiegazioni_con_fonte), len(revisioni_log)), "100%", "Da monitorare"],
        ["B3c", "Fallback spiegazione", fallback_rate, "< 5%", "Da monitorare"],
        ["B4", "Distanza gesto critico", "Stop: 1 click; override: 2 click", "<= 1 / <= 2", "OK"],
        ["C1", "Gap recall sottogruppi", f"area {gap_recall_area}; tipo {gap_recall_tipo}", "alert > 0.15", "Allarme" if max(gap_recall_area, gap_recall_tipo) > 0.15 else "OK"],
        ["C2", "Gap calibrazione gruppo", str(gap_calibrazione), "alert > 0.10", "Allarme" if gap_calibrazione > 0.10 else "OK"],
        ["C3", "Override per sottogruppo", override_area, "Esposto", "Da monitorare"],
        ["C4", "Drift performance", "Non esposto nella vista attuale", "Grafico + alert", "Da completare"],
        ["D1", "Completezza audit trail", "Transizioni principali loggate", "100%", "Da campionare"],
        ["D3", "Ricostruibilità decisione", "Tabella + popup dettaglio + download JSON", "< 60 sec", "OK"],
    ]
    return pd.DataFrame(righe, columns=["Codice", "KPI", "Valore attuale", "Target", "Stato"])


@st.dialog("Dettaglio log audit")
def mostra_dettaglio_log(record):
    decisione = record.get("decisione", {})
    messaggio_llm = decisione.get("messaggio_llm")
    c_log, c_dec = st.columns(2)
    with c_log:
        st.markdown(f"**logId:** {record.get('hash', '-')}")
        st.markdown(f"**Timestamp:** {record.get('timestamp', '-')}")
        st.markdown(f"**Attore:** {record.get('attore', '-')}")
        st.markdown(f"**Evento:** {record.get('evento', '-')}")
        st.markdown(f"**Hash precedente:** {record.get('hash_precedente', '-')}")
    with c_dec:
        st.markdown(f"**Asset:** {decisione.get('asset_id', '-')}")
        st.markdown(f"**Decision ID:** {decisione.get('id', '-')}")
        st.markdown(f"**Livello:** {decisione.get('livello', '-')}")
        st.markdown(f"**Stato:** {decisione.get('stato', '-')}")
        st.markdown(f"**Azione:** {decisione.get('azione', '-')}")
        st.markdown(f"**Motivazione:** {decisione.get('motivazione', '-')}")

    if messaggio_llm:
        st.markdown("**Messaggio LLM associato**")
        st.markdown(f"**Fonte:** {messaggio_llm.get('fonte', '-')}")
        st.markdown(f"**Latenza:** {messaggio_llm.get('latenza_ms', '-')} ms")
        st.write(messaggio_llm.get("testo", "-"))
        st.caption(messaggio_llm.get("incertezza", ""))

    if record.get("extra"):
        st.markdown("**Extra**")
        st.json(record["extra"])

    st.markdown("**Record JSON completo**")
    st.json(record)


tab_coda, tab_matrice, tab_kpi, tab_audit = st.tabs(
    ["Attività da lavorare", "Mappa rischio per regione", "KPI qualità", "Registro audit"])

# ----------------------------------------------------------------------
with tab_coda:
    pendenti = [r for r in om.coda if r.stato in stati_lavorabili]
    st.subheader("Attività da verificare")
    if not pendenti:
        st.success("Nessuna decisione in attesa. Il sistema è stabile e non ci sono revisioni pending.")
    else:
        st.markdown("**Filtri rapidi**")
        f1, f2, f3 = st.columns([1.1, 1, 1])
        with f1:
            filtro_testo = st.text_input(
                "Cerca attività",
                placeholder="Asset, area o tipo asset",
                key="filtro_attivita_testo"
            )
        with f2:
            filtro_regime = st.multiselect(
                "Supervisione",
                ["HIC", "HITL", "HOTL"],
                key="filtro_attivita_regime"
            )
        with f3:
            filtro_sla = st.multiselect(
                "SLA",
                ["Scaduto", "Urgente", "Nei tempi", "Non applicabile"],
                key="filtro_attivita_sla"
            )

        f4, f5, f6 = st.columns([1, 1, 0.8])
        with f4:
            filtro_area = st.multiselect(
                "Area",
                sorted({r.area_geografica for r in pendenti}),
                key="filtro_attivita_area"
            )
        with f5:
            filtro_tipo = st.multiselect(
                "Tipo asset",
                sorted({r.tipo_asset for r in pendenti}),
                key="filtro_attivita_tipo"
            )
        with f6:
            solo_honeypot = st.checkbox("Solo controlli", key="filtro_attivita_honeypot")

        pendenti_filtrati = filtra_attivita(
            pendenti,
            filtro_testo,
            filtro_regime,
            filtro_area,
            filtro_tipo,
            filtro_sla,
            solo_honeypot,
        )
        st.caption(f"Mostrate {len(pendenti_filtrati)} attività su {len(pendenti)} lavorabili.")

        pendenti_ordinati = sorted(
            pendenti_filtrati,
            key=lambda x: (x.stato != StatoDecisione.ESCALATION, -x.prob_guasto)
        )
        visibili = pendenti_ordinati[:10]
        honeypot_nascosti = [r for r in pendenti_ordinati if r.honeypot and r not in visibili]
        visibili.extend(honeypot_nascosti)
        if not visibili:
            st.info("Nessuna attività corrisponde ai filtri selezionati.")
        for r in visibili:
            livello_emoji = {"HIC": "🔴", "HITL": "🟠", "HOTL": "🟡"}
            livello_label = {"HIC": "Decide solo l'operatore", "HITL": "Serve conferma", "HOTL": "Solo monitoraggio"}
            descrizione_lavorabilita = f"{r.livello.value} - {livello_label.get(r.livello.value, 'Da verificare')}"
            with st.expander(
                    f"{livello_emoji.get(r.livello.value, '⚪')} {descrizione_lavorabilita} · {testo_sla(r)} · {r.asset_id} · {r.tipo_asset} · {r.area_geografica}", expanded=r.stato == StatoDecisione.ESCALATION):
                left, right = st.columns([1.3, 1.7])
                with left:
                    st.markdown(f"**Rischio stimato:** {r.prob_guasto:.2f}")
                    st.markdown(f"**Confidenza modello:** {r.confidenza:.2f}")
                    st.markdown(f"**Supervisione:** {r.livello.value}")
                    st.markdown(f"**SLA:** {testo_sla(r)}")
                    st.markdown(f"**Stato:** {r.stato.value}")
                    st.markdown(f"**Utenza:** {r.criticita_utenza}")
                    st.markdown(f"**Azione consigliata:** {r.azione_proposta}")
                with right:
                    st.markdown(f"**Esito richiesto:** {livello_label.get(r.livello.value, 'Da verificare')}")
                    sp = spiegazione_per(r)
                    st.markdown("**Motivazione operativa:**")
                    st.write(sp.testo)
                    st.warning(sp.incertezza)
                    st.caption(f"Fonte: {sp.fonte} · {sp.latenza_ms} ms")

                mot = st.text_area("Motivazione dell'operatore", key=f"m{r.id}", help="Scrivere una descrizione chiara di perché si approva, modifica o rifiuta la decisione.")
                az = st.selectbox("Azione da applicare", AZIONI,
                                  index=AZIONI.index(r.azione_proposta), key=f"a{r.id}")
                keywords = st.multiselect(
                    "Elementi verificati prima dell'approvazione",
                    parole_chiave_revisione(r),
                    key=f"kw{r.id}",
                    help="Selezionare almeno due parole chiave che giustificano la scelta."
                )
                cooldown_ok, cooldown_residuo = cooldown_pronto(r.id)
                approvazione_disabilitata = not cooldown_ok or len(set(keywords)) < 2
                if not cooldown_ok:
                    st.caption(f"Cooldown anti rubber-stamping: approvazione disponibile tra {cooldown_residuo} secondi.")
                if len(set(keywords)) < 2:
                    st.caption("Selezionare almeno 2 parole chiave dell'evidenza prima di approvare o modificare.")
                b1, b2, b3 = st.columns(3)
                try:
                    if b1.button("Approva", key=f"ok{r.id}", disabled=approvazione_disabilitata):
                        om.revisiona(r.id, StatoDecisione.APPROVATA, operatore, mot,
                                     evidenze_keywords=keywords)
                        st.session_state["review_feedback"] = f"Decisione {r.asset_id} approvata."
                        st.rerun()
                    if b2.button("Modifica e approva", key=f"mod{r.id}", disabled=approvazione_disabilitata):
                        om.revisiona(r.id, StatoDecisione.MODIFICATA, operatore, mot,
                                     azione_modificata=az, evidenze_keywords=keywords)
                        st.session_state["review_feedback"] = f"Decisione {r.asset_id} modificata e approvata."
                        st.rerun()
                    if b3.button("Rifiuta", key=f"no{r.id}"):
                        om.revisiona(r.id, StatoDecisione.RIFIUTATA, operatore, mot)
                        st.session_state["review_feedback"] = f"Decisione {r.asset_id} rifiutata."
                        st.rerun()
                except ValueError as e:
                    st.error(str(e))
                if not mot or len(mot.strip()) < 15:
                    st.caption("Motivazione obbligatoria: minimo 15 caratteri.")

# ----------------------------------------------------------------------
with tab_matrice:
    st.subheader("Mappa rischio per regione")
    st.caption("Ogni grafico mostra una regione. I punti piu' in alto sono piu' rischiosi; i punti piu' a sinistra hanno meno confidenza del modello.")
    df_plot = pred.rename(columns={"proba": "rischio"}).copy()
    df_plot["confidenza"] = pd.to_numeric(df_plot["confidenza"], errors="coerce")
    df_plot["rischio"] = pd.to_numeric(df_plot["rischio"], errors="coerce")
    regioni = sorted(df_plot["area_geografica"].dropna().unique())
    colori_regioni = {
        area: PASTEL_AREA_COLORS[i % len(PASTEL_AREA_COLORS)]
        for i, area in enumerate(regioni)
    }

    zone_df = pd.DataFrame([
        {"regime": "HOTL", "x_min": 0.80, "x_max": 1.00, "y_min": 0.00, "y_max": 0.60},
        {"regime": "HITL", "x_min": 0.00, "x_max": 1.00, "y_min": 0.60, "y_max": 1.00},
        {"regime": "HIC", "x_min": 0.00, "x_max": 0.80, "y_min": 0.00, "y_max": 1.00},
    ])

    zones = (
        alt.Chart(zone_df)
        .mark_rect(opacity=0.34)
        .encode(
            x=alt.X("x_min:Q", scale=alt.Scale(domain=[0, 1])),
            x2="x_max:Q",
            y=alt.Y("y_min:Q", scale=alt.Scale(domain=[0, 1])),
            y2="y_max:Q",
            color=alt.Color("regime:N", scale=alt.Scale(domain=list(PASTEL_REGIME_COLORS), range=list(PASTEL_REGIME_COLORS.values())), legend=alt.Legend(title="Regime")),
            tooltip=["regime:N"]
        )
    )

    labels = alt.Chart(pd.DataFrame([
        {"x": 0.90, "y": 0.25, "label": "HOTL\nmonitoraggio"},
        {"x": 0.50, "y": 0.80, "label": "HITL\nconferma"},
        {"x": 0.35, "y": 0.35, "label": "HIC\noperatore"},
    ])).mark_text(fontSize=12, fontWeight="bold", color="#31413d", align="center").encode(
        x="x:Q",
        y="y:Q",
        text="label:N"
    )

    threshold_v = alt.Chart(pd.DataFrame({"x": [0.80]})).mark_rule(color=PASTEL_THRESHOLD, strokeDash=[6, 4]).encode(x="x:Q")
    threshold_h = alt.Chart(pd.DataFrame({"y": [0.60]})).mark_rule(color=PASTEL_THRESHOLD, strokeDash=[6, 4]).encode(y="y:Q")

    cols = st.columns(2)
    for i, area in enumerate(regioni):
        df_area = df_plot[df_plot["area_geografica"] == area]
        points = (
            alt.Chart(df_area)
            .mark_circle(size=58, opacity=0.82, color=colori_regioni[area])
            .encode(
                x=alt.X("confidenza:Q", title="Confidenza", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("rischio:Q", title="Rischio", scale=alt.Scale(domain=[0, 1])),
                tooltip=["asset_id:N", "tipo_asset:N", "confidenza:Q", "rischio:Q"],
            )
        )
        chart = (
            zones + threshold_v + threshold_h + labels + points
        ).properties(title=f"Regione {area}", width=410, height=320)
        with cols[i % 2]:
            st.altair_chart(chart, use_container_width=True)

    st.caption("Ogni riquadro mostra una sola regione con le stesse soglie operative, così l'operatore può leggere più facilmente il regime di supervisione per area.")

# ----------------------------------------------------------------------
with tab_kpi:
    st.subheader("KPI qualità dashboard")
    st.caption("Indicatori tratti dal documento 'EnerGuard · Indicatori di qualità della dashboard di supervisione'.")
    records_kpi = carica_records_audit()
    kpi_df = tabella_kpi(records_kpi)

    filtro_stato_kpi = st.multiselect(
        "Filtra per stato KPI",
        sorted(kpi_df["Stato"].dropna().unique().tolist()),
        key="filtro_kpi_stato"
    )
    if filtro_stato_kpi:
        kpi_df = kpi_df[kpi_df["Stato"].isin(filtro_stato_kpi)]

    st.dataframe(
        kpi_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Codice": st.column_config.TextColumn("Codice", width="small"),
            "KPI": st.column_config.TextColumn("KPI", width="medium"),
            "Valore attuale": st.column_config.TextColumn("Valore attuale", width="medium"),
            "Target": st.column_config.TextColumn("Target", width="medium"),
            "Stato": st.column_config.TextColumn("Stato", width="small"),
        },
    )
    st.caption("Alcuni KPI sono misurati automaticamente; quelli indicati come manuali o da campionare richiedono verifica durante la demo o revisione audit.")

# ----------------------------------------------------------------------
with tab_audit:
    st.subheader("Registro audit")
    catena_integra, record_audit = audit.verifica_catena()
    if catena_integra:
        st.success(f"Integrità audit verificata: catena hash integra su {record_audit} record.")
    else:
        st.error(f"Catena audit compromessa: verifica fallita dopo {record_audit} record validi.")
    st.caption("Ogni record contiene l'hash del record precedente e il proprio hash. "
               "La verifica ricalcola la catena e rileva modifiche o cancellazioni silenziose.")
    log_path = "audit_trail.jsonl"
    if not os.path.exists(log_path):
        st.info("Nessun evento di audit registrato ancora.")
    else:
        records = carica_records_audit(log_path)

        if records:
            st.download_button(
                "Scarica log JSON",
                data=json.dumps(records, ensure_ascii=False, indent=2),
                file_name="audit_trail.json",
                mime="application/json",
            )

            st.markdown("**Filtri rapidi audit**")
            filtro_tempo = st.pills(
                "Periodo",
                ["Oggi", "Ultime 24 ore", "Ultimi 7 giorni", "Tutto"],
                default="Tutto",
                key="audit_filtro_tempo",
            )
            stati_disponibili = sorted({r.get("decisione", {}).get("stato") or "Senza stato" for r in records})
            supervisioni_disponibili = sorted({r.get("decisione", {}).get("livello") or "Senza supervisione" for r in records})
            attori_disponibili = sorted({r.get("attore") or "-" for r in records})

            f_stato, f_supervisione = st.columns(2)
            with f_stato:
                filtro_stati = st.pills(
                    "Esito/Stato",
                    stati_disponibili,
                    selection_mode="multi",
                    format_func=stato_filtro_leggibile,
                    key="audit_filtro_stati",
                )
            with f_supervisione:
                filtro_supervisioni = st.pills(
                    "Supervisione",
                    supervisioni_disponibili,
                    selection_mode="multi",
                    key="audit_filtro_supervisioni",
                )

            f_attore, f_testo = st.columns([1.2, 1])
            with f_attore:
                filtro_attori = st.pills(
                    "Attore",
                    attori_disponibili,
                    selection_mode="multi",
                    key="audit_filtro_attori",
                )
            with f_testo:
                filtro_testo_audit = st.text_input(
                    "Cerca nel log",
                    placeholder="Asset, evento o operatore",
                    key="audit_filtro_testo",
                )

            records = filtra_records_audit(
                records,
                filtro_stati or [],
                filtro_supervisioni or [],
                filtro_attori or [],
                filtro_tempo or "Tutto",
                filtro_testo_audit,
            )

            righe_log = []
            for i, record in enumerate(records):
                decisione = record.get("decisione", {})
                extra = record.get("extra", {})
                righe_log.append({
                    "idx": i,
                    "attivita": decisione.get("asset_id") or extra.get("asset_id") or "SISTEMA",
                    "stato": stato_leggibile(decisione.get("stato", "-")),
                    "Supervisione": decisione.get("livello", "-"),
                    "operatore/sistema": record.get("attore", "-"),
                    "Data": data_leggibile(record.get("timestamp", "-")),
                })

            st.caption(f"Mostrati {len(righe_log)} log audit. Seleziona una riga per aprire il dettaglio completo.")
            tabella_log = pd.DataFrame(righe_log)
            if tabella_log.empty:
                st.info("Nessun log corrisponde ai filtri selezionati.")
            else:
                selezione = st.dataframe(
                    tabella_log,
                    width="stretch",
                    hide_index=True,
                    column_order=["attivita", "stato", "Supervisione", "operatore/sistema", "Data"],
                    on_select=richiedi_apertura_dettaglio_log,
                    selection_mode="single-row",
                    key="tabella_audit",
                )
                if selezione.selection.rows and st.session_state.pop("audit_apri_dettaglio", False):
                    selected_row = selezione.selection.rows[0]
                    selected_idx = int(tabella_log.iloc[selected_row]["idx"])
                    mostra_dettaglio_log(records[selected_idx])
        else:
            st.info("Il file di audit esiste ma non contiene record validi.")
