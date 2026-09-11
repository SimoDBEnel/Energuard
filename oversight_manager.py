"""
EnerGuard Starter Kit - OversightManager
=========================================
Gestisce il routing delle raccomandazioni AI verso i tre livelli di
supervisione (HIC / HITL / HOTL) e la coda di approvazione umana.

REGOLA D'ORO: nessuna azione con esito "IN_ATTESA" puo' essere eseguita.
Il vostro sistema deve dimostrarlo (la giuria lo testera' dal vivo).

TODO per il team:
  1. Completare la matrice di routing in `route()` giustificando le soglie
     nella Dichiarazione di Oversight (deliverable D3).
  2. Implementare l'escalation: una decisione HITL non revisionata entro
     `sla_minuti` deve cambiare stato, non essere eseguita in silenzio.
  3. Collegare ogni transizione di stato all'AuditLogger.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Optional
import uuid


class LivelloSupervisione(str, Enum):
    HOTL = "HOTL"   # il sistema agisce, l'umano monitora ex post
    HITL = "HITL"   # approvazione umana obbligatoria prima dell'esecuzione
    HIC = "HIC"     # veto umano assoluto: solo l'umano decide


class StatoDecisione(str, Enum):
    IN_ATTESA = "IN_ATTESA"
    APPROVATA = "APPROVATA"
    MODIFICATA = "MODIFICATA"
    RIFIUTATA = "RIFIUTATA"
    ESCALATION = "ESCALATION"
    AUTO_ESEGUITA = "AUTO_ESEGUITA"   # ammessa SOLO per HOTL
    BLOCCATA_STOP = "BLOCCATA_STOP"   # bloccata da emergency stop


AZIONI = ["nessuna_azione", "ispezione_routine", "programma_manutenzione",
          "riduci_carico", "ispezione_urgente"]


@dataclass
class Raccomandazione:
    asset_id: str
    tipo_asset: str
    area_geografica: str
    criticita_utenza: str          # standard | alta | critica
    prob_guasto: float             # output del modello, 0-1
    confidenza: float              # 0-1 (vedi nota in fondo)
    azione_proposta: str
    spiegazione: list              # es. [("vibrazione_indice", 0.41), ...]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    creata_il: str = field(default_factory=lambda: datetime.now().isoformat())
    livello: Optional[LivelloSupervisione] = None
    stato: StatoDecisione = StatoDecisione.IN_ATTESA
    revisore: Optional[str] = None
    motivazione: Optional[str] = None
    revisionata_il: Optional[str] = None
    evidenze_keywords: list[str] = field(default_factory=list)
    honeypot: bool = False
    honeypot_messaggio: Optional[str] = None
    messaggio_llm: Optional[dict] = None


class OversightManager:
    def __init__(self, audit_logger, soglia_confidenza_alta: float = 0.80,
                 soglia_rischio_alto: float = 0.60, sla_minuti: int = 30,
                 stop_state_path: Optional[str] = None):
        self.audit = audit_logger
        self.soglia_conf = soglia_confidenza_alta
        self.soglia_rischio = soglia_rischio_alto
        self.sla_minuti = sla_minuti
        self.coda: list[Raccomandazione] = []
        self.stop_state_path = Path(stop_state_path) if stop_state_path else None
        self.stop_gruppi: list[frozenset[str]] = self._carica_stop()
        self.mitigazione_prudenziale = False
        self.motivi_mitigazione: list[str] = []
        self._storico_motivazioni: list[tuple[datetime, str]] = []

    @property
    def stop_attivi(self) -> set[str]:
        return {ambito for gruppo in self.stop_gruppi for ambito in gruppo}

    def _carica_stop(self) -> list[frozenset[str]]:
        if self.stop_state_path is None or not self.stop_state_path.exists():
            return []
        try:
            dati = json.loads(self.stop_state_path.read_text(encoding="utf-8"))
            gruppi = dati.get("stop_gruppi")
            if gruppi is None:
                gruppi = [[ambito] for ambito in dati.get("stop_attivi", [])]
            return [frozenset(str(ambito) for ambito in gruppo if str(ambito).strip())
                    for gruppo in gruppi if gruppo]
        except (OSError, json.JSONDecodeError, AttributeError):
            return []

    def _salva_stop(self):
        if self.stop_state_path is None:
            return
        self.stop_state_path.write_text(
            json.dumps({"stop_gruppi": [sorted(gruppo) for gruppo in self.stop_gruppi]},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _normalizza_motivazione(motivazione: str) -> str:
        return " ".join(motivazione.strip().split()).casefold()

    def _valida_motivazione(self, motivazione: str, *, consentire_duplicati: bool = False):
        if motivazione is None:
            raise ValueError("Motivazione obbligatoria (minimo 15 caratteri).")
        testo = motivazione.strip()
        if len(testo) < 15:
            raise ValueError("Motivazione obbligatoria (minimo 15 caratteri). "
                             "Senza motivazione non c'e' audit trail.")
        if len(testo.split()) <= 2:
            raise ValueError("Motivazione obbligatoria: inserire piu' di 2 parole.")
        if not consentire_duplicati:
            normalized = self._normalizza_motivazione(testo)
            limite = datetime.now() - timedelta(minutes=30)
            recenti = [m for ts, m in self._storico_motivazioni if ts >= limite]
            if any(self._normalizza_motivazione(m) == normalized for m in recenti):
                raise ValueError("Motivazione duplicata negli ultimi 30 minuti. "
                                 "Inserire un testo originale e specifico.")
        return testo

    @staticmethod
    def _valida_evidenze_keywords(esito: StatoDecisione, evidenze_keywords: Optional[list[str]]):
        if esito not in (StatoDecisione.APPROVATA, StatoDecisione.MODIFICATA):
            return []
        keywords = [k.strip() for k in (evidenze_keywords or []) if k and k.strip()]
        if len(set(keywords)) < 2:
            raise ValueError("Prima di approvare devi selezionare almeno 2 parole chiave "
                             "dell'evidenza esaminata.")
        return keywords

    # ------------------------------------------------------------------
    # ROUTING - il cuore dell'esercizio
    # ------------------------------------------------------------------
    def route(self, r: Raccomandazione) -> LivelloSupervisione:
        """Assegna il livello di supervisione. COMPLETARE E GIUSTIFICARE.

        Logica minima di partenza (da estendere):
          - utenza critica (ospedali, infrastrutture) o azione "riduci_carico"
            su utenza alta/critica  -> HIC, sempre
          - rischio alto O confidenza bassa                     -> HITL
          - rischio basso E confidenza alta E azione leggera    -> HOTL
        """
        if r.criticita_utenza == "critica" or (
                r.azione_proposta == "riduci_carico" and r.criticita_utenza != "standard"):
            livello = LivelloSupervisione.HIC
        elif r.prob_guasto >= self.soglia_rischio or r.confidenza < self.soglia_conf:
            livello = LivelloSupervisione.HITL
        elif r.azione_proposta in ("nessuna_azione", "ispezione_routine"):
            livello = LivelloSupervisione.HOTL
        else:
            livello = LivelloSupervisione.HITL   # default prudente
        if livello == LivelloSupervisione.HOTL and self.mitigazione_prudenziale:
            livello = LivelloSupervisione.HITL
        r.livello = livello
        return livello

    def configura_mitigazione_prudenziale(self, attiva: bool, motivi: Iterable[str],
                                          attore: str = "SISTEMA"):
        motivi_norm = [str(motivo).strip() for motivo in motivi if str(motivo).strip()]
        if self.mitigazione_prudenziale == attiva and self.motivi_mitigazione == motivi_norm:
            return False
        self.mitigazione_prudenziale = attiva
        self.motivi_mitigazione = motivi_norm if attiva else []
        self.audit.log(attore, "mitigazione_prudenziale_ON" if attiva
                       else "mitigazione_prudenziale_OFF", None,
                       extra={"motivi": motivi_norm,
                              "effetto": "promozione HOTL a HITL"})
        return True

    # ------------------------------------------------------------------
    def sottometti(self, r: Raccomandazione):
        """Instrada la raccomandazione e la esegue o la mette in coda."""
        if self._stop_applicabile(r):
            r.stato = StatoDecisione.BLOCCATA_STOP
            self.audit.log("SISTEMA", "blocco_emergency_stop", r)
            return r

        self.route(r)
        if r.livello == LivelloSupervisione.HOTL:
            r.stato = StatoDecisione.AUTO_ESEGUITA
            self._esegui(r)
            self.audit.log("SISTEMA", "auto_esecuzione_HOTL", r)
        else:
            self.coda.append(r)
            self.audit.log("SISTEMA", f"in_coda_{r.livello.value}", r)
        return r

    def revisiona(self, decision_id: str, esito: StatoDecisione,
                  revisore: str, motivazione: str,
                  azione_modificata: Optional[str] = None,
                  evidenze_keywords: Optional[list[str]] = None):
        """Registra il giudizio umano. La motivazione e' OBBLIGATORIA."""
        testo = self._valida_motivazione(motivazione)
        keywords = self._valida_evidenze_keywords(esito, evidenze_keywords)
        r = self._trova(decision_id)
        if r.stato not in (StatoDecisione.IN_ATTESA, StatoDecisione.ESCALATION):
            raise ValueError(f"Decisione {decision_id} gia' chiusa: {r.stato}")
        if r.honeypot and esito == StatoDecisione.APPROVATA:
            r.revisore, r.motivazione, r.evidenze_keywords = revisore, testo, keywords
            self.audit.log(revisore, "allerta_honeypot_approvato", r,
                           extra={"messaggio": r.honeypot_messaggio})
            self._storico_motivazioni.append((datetime.now(), testo))
            raise ValueError("ALLERTA RUBBER STAMPING: questa raccomandazione era "
                             "un controllo honeypot palesemente incoerente.")
        r.stato, r.revisore, r.motivazione = esito, revisore, testo
        r.revisionata_il = datetime.now().isoformat()
        r.evidenze_keywords = keywords
        self._storico_motivazioni.append((datetime.now(), testo))
        if azione_modificata:
            r.azione_proposta = azione_modificata
        if esito in (StatoDecisione.APPROVATA, StatoDecisione.MODIFICATA):
            self._esegui(r)
        self.audit.log(revisore, f"revisione_{esito.value}", r)
        return r

    def minuti_residui_sla(self, r: Raccomandazione, adesso: Optional[datetime] = None) -> Optional[int]:
        if r.livello != LivelloSupervisione.HITL:
            return None
        if r.stato not in (StatoDecisione.IN_ATTESA, StatoDecisione.ESCALATION):
            return None
        adesso = adesso or datetime.now()
        creata = datetime.fromisoformat(r.creata_il)
        scadenza = creata + timedelta(minutes=self.sla_minuti)
        return int((scadenza - adesso).total_seconds() // 60)

    def aggiorna_sla(self, adesso: Optional[datetime] = None) -> int:
        adesso = adesso or datetime.now()
        escalate = 0
        for r in self.coda:
            minuti_residui = self.minuti_residui_sla(r, adesso)
            if r.stato == StatoDecisione.IN_ATTESA and minuti_residui is not None and minuti_residui < 0:
                r.stato = StatoDecisione.ESCALATION
                self.audit.log("SISTEMA", "escalation_sla_scaduto", r,
                               extra={"sla_minuti": self.sla_minuti,
                                      "minuti_ritardo": abs(minuti_residui)})
                escalate += 1
        return escalate

    # ------------------------------------------------------------------
    # EMERGENCY STOP - deve bloccare DAVVERO (la giuria lo verifica)
    # ------------------------------------------------------------------
    def attiva_stop(self, ambito: str, operatore: str, motivazione: str):
        """ambito: 'GLOBALE' | 'area:<nome>' | 'tipo:<tipo_asset>'"""
        return self.attiva_stop_filtri([ambito], operatore, motivazione)

    def attiva_stop_filtri(self, ambiti: Iterable[str], operatore: str, motivazione: str):
        """Attiva uno stop su uno o piu' ambiti e blocca la coda gia' esposta."""
        testo = self._valida_motivazione(motivazione, consentire_duplicati=True)
        ambiti_norm = self._normalizza_ambiti(ambiti)
        gruppo = frozenset(ambiti_norm)
        if "GLOBALE" in gruppo and len(gruppo) > 1:
            raise ValueError("Lo stop globale non puo' essere combinato con filtri specifici.")
        if "GLOBALE" in gruppo:
            self.stop_gruppi = [gruppo]
        elif frozenset({"GLOBALE"}) in self.stop_gruppi:
            raise ValueError("Disattivare lo stop globale prima di attivare filtri specifici.")
        elif gruppo not in self.stop_gruppi:
            self.stop_gruppi.append(gruppo)
        self._salva_stop()
        decisioni_bloccate = self._blocca_decisioni_in_stop()
        self.audit.log(operatore, "emergency_stop_ON", None,
                       extra={"gruppo": ambiti_norm, "semantica": "OR intra-dimensione; AND inter-dimensione",
                              "motivazione": testo, "decisioni_bloccate": decisioni_bloccate})
        self._storico_motivazioni.append((datetime.now(), testo))
        return decisioni_bloccate

    def disattiva_stop(self, ambito: str, operatore: str, motivazione: str):
        return self.disattiva_stop_filtri([ambito], operatore, motivazione)

    def disattiva_stop_filtri(self, ambiti: Iterable[str], operatore: str, motivazione: str):
        testo = self._valida_motivazione(motivazione, consentire_duplicati=True)
        ambiti_norm = self._normalizza_ambiti(ambiti)
        gruppo = frozenset(ambiti_norm)
        if gruppo not in self.stop_gruppi:
            raise ValueError("Il gruppo di stop selezionato non e' attivo.")
        self.stop_gruppi.remove(gruppo)
        self._salva_stop()
        decisioni_ripristinate = self._ripristina_decisioni_fuori_stop()
        self.audit.log(operatore, "emergency_stop_OFF", None,
                       extra={"gruppo": ambiti_norm, "motivazione": testo,
                              "decisioni_ripristinate": decisioni_ripristinate})
        self._storico_motivazioni.append((datetime.now(), testo))
        return decisioni_ripristinate

    @staticmethod
    def _normalizza_ambiti(ambiti: Iterable[str]) -> list[str]:
        normalizzati = []
        for ambito in ambiti:
            testo = str(ambito).strip()
            if testo and testo not in normalizzati:
                normalizzati.append(testo)
        if not normalizzati:
            raise ValueError("Selezionare almeno un filtro per l'emergency stop.")
        return normalizzati

    def _stop_applicabile(self, r: Raccomandazione) -> bool:
        for gruppo in self.stop_gruppi:
            if "GLOBALE" in gruppo:
                return True
            aree = {ambito.removeprefix("area:") for ambito in gruppo if ambito.startswith("area:")}
            tipi = {ambito.removeprefix("tipo:") for ambito in gruppo if ambito.startswith("tipo:")}
            if ((not aree or r.area_geografica in aree)
                    and (not tipi or r.tipo_asset in tipi)):
                return True
        return False

    def _blocca_decisioni_in_stop(self) -> int:
        bloccate = 0
        for r in self.coda:
            if r.stato == StatoDecisione.IN_ATTESA and self._stop_applicabile(r):
                r.stato = StatoDecisione.BLOCCATA_STOP
                self.audit.log("SISTEMA", "blocco_emergency_stop_attivato", r,
                               extra={"stop_gruppi": [sorted(g) for g in self.stop_gruppi]})
                bloccate += 1
        return bloccate

    def _ripristina_decisioni_fuori_stop(self) -> int:
        ripristinate = 0
        for r in self.coda:
            if r.stato == StatoDecisione.BLOCCATA_STOP and not self._stop_applicabile(r):
                r.stato = StatoDecisione.IN_ATTESA
                self.audit.log("SISTEMA", "ripristino_post_emergency_stop", r,
                               extra={"stop_gruppi": [sorted(g) for g in self.stop_gruppi]})
                ripristinate += 1
        return ripristinate


    # ------------------------------------------------------------------
    def _esegui(self, r: Raccomandazione):
        """Punto unico di esecuzione. Qualunque azione DEVE passare da qui.
        La giuria verifichera' che non esistano altri percorsi di esecuzione."""
        assert r.stato in (StatoDecisione.APPROVATA, StatoDecisione.MODIFICATA,
                           StatoDecisione.AUTO_ESEGUITA), \
            f"Tentata esecuzione con stato non valido: {r.stato}"
        assert not (r.livello == LivelloSupervisione.HIC
                    and r.stato == StatoDecisione.AUTO_ESEGUITA), \
            "Violazione: una decisione HIC non puo' mai essere auto-eseguita"
        print(f"[ESECUZIONE] {r.asset_id}: {r.azione_proposta} ({r.livello.value})")

    def _trova(self, decision_id: str) -> Raccomandazione:
        for r in self.coda:
            if r.id == decision_id:
                return r
        raise KeyError(decision_id)

    # ------------------------------------------------------------------
    # KPI per il pannello di monitoraggio (vedi Indicatori di Qualita')
    # ------------------------------------------------------------------
    def kpi(self) -> dict:
        chiuse = [r for r in self.coda if r.stato in (
            StatoDecisione.APPROVATA, StatoDecisione.MODIFICATA,
            StatoDecisione.RIFIUTATA)]
        override = [r for r in chiuse if r.stato in
                    (StatoDecisione.RIFIUTATA, StatoDecisione.MODIFICATA)]
        durate = []
        for r in chiuse:
            if not r.revisionata_il:
                continue
            durata = (datetime.fromisoformat(r.revisionata_il)
                       - datetime.fromisoformat(r.creata_il)).total_seconds() / 60
            durate.append(durata)
        distribuzione = {
            livello.value: sum(1 for r in self.coda if r.livello == livello)
            for livello in LivelloSupervisione
        }
        motivazioni = [r.motivazione.strip() for r in chiuse if r.motivazione]
        motivazioni_brevi = sum(1 for testo in motivazioni if len(testo) < 30)
        return {
            "in_attesa": sum(1 for r in self.coda if r.stato == StatoDecisione.IN_ATTESA),
            "tasso_override": round(len(override) / len(chiuse), 3) if chiuse else None,
            "stop_attivi": sorted(self.stop_attivi),
            "tempo_medio_revisione_min": round(mean(durate), 2) if durate else None,
            "tempo_mediano_revisione_min": round(median(durate), 2) if durate else None,
            "distribuzione_livelli": distribuzione,
            "motivazioni_brevi_pct": round(100 * motivazioni_brevi / len(motivazioni), 2) if motivazioni else None,
        }


# NOTA SU "CONFIDENZA": per un classificatore binario una scelta semplice e
# difendibile e' confidenza = max(p, 1-p) oppure 1 - entropia normalizzata.
# Qualunque scelta va dichiarata nella model card.
