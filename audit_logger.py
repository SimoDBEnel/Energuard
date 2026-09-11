"""
EnerGuard Starter Kit - AuditLogger
====================================
Log append-only in JSONL con concatenazione di hash: ogni riga contiene
l'hash della precedente, quindi qualunque manomissione a posteriori rompe
la catena ed e' rilevabile. Risponde ai requisiti di tracciabilita'
(Art. 12 e Art. 14 AI Act).

TODO per il team:
  1. Loggare TUTTE le transizioni di stato, non solo le approvazioni:
     ingresso in coda, auto-esecuzioni HOTL, escalation, stop, sblocchi.
  2. Esporre nella dashboard la verifica di integrita' (`verifica_catena`).
  3. Aggiungere un filtro/ricerca del log per asset, operatore, periodo.
"""

import hashlib
import json
import os
import fcntl
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class AuditLogger:
    def __init__(self, percorso: str = "audit_trail.jsonl"):
        self.path = Path(percorso)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            integra, validi = self._verifica_catena_senza_lock()
            if self.path.exists() and not integra:
                self._archivia_catena_compromessa(validi)

    def log(self, attore: str, evento: str, raccomandazione=None, extra: dict = None):
        with self._lock():
            record = {
                "timestamp": datetime.now().isoformat(),
                "attore": attore,
                "evento": evento,
                "hash_precedente": self._recupera_ultimo_hash(),
            }
            if raccomandazione is not None:
                record["decisione"] = {
                    "id": raccomandazione.id,
                    "asset_id": raccomandazione.asset_id,
                    "livello": getattr(raccomandazione.livello, "value", None),
                    "stato": raccomandazione.stato.value,
                    "azione": raccomandazione.azione_proposta,
                    "prob_guasto": raccomandazione.prob_guasto,
                    "confidenza": raccomandazione.confidenza,
                    "motivazione": raccomandazione.motivazione,
                    "spiegazione": raccomandazione.spiegazione,
                }
                if getattr(raccomandazione, "messaggio_llm", None):
                    record["decisione"]["messaggio_llm"] = raccomandazione.messaggio_llm
            if extra:
                record["extra"] = extra
            record["hash"] = self._hash(record)
            self._append(record)
        return record

    def verifica_catena(self) -> tuple[bool, int]:
        """Ritorna (integra?, n_record). Da mostrare nella dashboard."""
        with self._lock():
            return self._verifica_catena_senza_lock()

    def _verifica_catena_senza_lock(self) -> tuple[bool, int]:
        precedente = "GENESI"
        n = 0
        for riga in self._leggi():
            atteso = riga.pop("hash")
            if riga.get("hash_precedente") != precedente or self._hash(riga) != atteso:
                return False, n
            precedente = atteso
            n += 1
        return True, n

    def _leggi(self):
        if not self.path.exists():
            return []
        return [json.loads(r) for r in self.path.read_text(encoding="utf-8").splitlines()]

    def _recupera_ultimo_hash(self) -> str:
        record = self._leggi()
        return record[-1]["hash"] if record else "GENESI"

    @contextmanager
    def _lock(self):
        with self.lock_path.open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _append(self, record: dict):
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def _archivia_catena_compromessa(self, record_validi: int):
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        archivio = self.path.with_name(f"{self.path.stem}.corrotto-{timestamp}{self.path.suffix}")
        totale = len(self._leggi())
        self.path.replace(archivio)
        record = {
            "timestamp": datetime.now().isoformat(),
            "attore": "SISTEMA",
            "evento": "migrazione_catena_audit_compromessa",
            "hash_precedente": "GENESI",
            "extra": {
                "archivio": archivio.name,
                "record_validi": record_validi,
                "record_totali": totale,
                "sha256_archivio": hashlib.sha256(archivio.read_bytes()).hexdigest(),
            },
        }
        record["hash"] = self._hash(record)
        self._append(record)

    @staticmethod
    def _hash(record: dict) -> str:
        return hashlib.sha256(
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
