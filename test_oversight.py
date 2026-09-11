import unittest
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_manager import (LivelloSupervisione, OversightManager, Raccomandazione,
                               StatoDecisione, metriche_rubber_stamping)
from audit_logger import AuditLogger


class TestOversightValidation(unittest.TestCase):
    def _raccomandazione(self, asset_id="A-1", *, honeypot=False):
        return Raccomandazione(
            asset_id=asset_id,
            tipo_asset="trasformatore",
            area_geografica="Nord",
            criticita_utenza="standard",
            prob_guasto=0.8,
            confidenza=0.9,
            azione_proposta="programma_manutenzione",
            spiegazione=[],
            honeypot=honeypot,
            honeypot_messaggio="Caso incoerente di controllo" if honeypot else None,
        )

    def test_motivazione_vuota_o_troppo_breve(self):
        om = OversightManager(AuditLogger("test_audit_temp.jsonl"))
        r = self._raccomandazione()
        om.coda.append(r)
        with self.assertRaises(ValueError):
            om.revisiona(r.id, StatoDecisione.APPROVATA, "OP-01", "breve")
        with self.assertRaises(ValueError):
            om.revisiona(r.id, StatoDecisione.APPROVATA, "OP-01", "")

    def test_motivazione_duplicata_tra_5_precedenti(self):
        om = OversightManager(AuditLogger("test_audit_temp.jsonl"))
        r = self._raccomandazione("A-2")
        om.coda.append(r)
        repeated = "Motivazione valida per controllo umano"
        om._storico_motivazioni.append((datetime.now() - timedelta(minutes=10), repeated))
        with self.assertRaises(ValueError):
            om.revisiona(r.id, StatoDecisione.APPROVATA, "OP-01", repeated,
                         evidenze_keywords=["rischio", "confidenza"])

    def test_approvazione_richiede_evidenze_keywords(self):
        om = OversightManager(AuditLogger("test_audit_temp.jsonl"))
        r = self._raccomandazione("A-3")
        om.coda.append(r)
        with self.assertRaises(ValueError):
            om.revisiona(r.id, StatoDecisione.APPROVATA, "OP-01",
                         "Motivazione valida e specifica", evidenze_keywords=["rischio"])

    def test_honeypot_approvato_genera_allerta(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            om = OversightManager(AuditLogger(str(path)))
            r = self._raccomandazione("A-4", honeypot=True)
            om.coda.append(r)
            with self.assertRaises(ValueError):
                om.revisiona(r.id, StatoDecisione.APPROVATA, "OP-01",
                             "Motivazione valida ma errata", evidenze_keywords=["rischio", "confidenza"])
            self.assertEqual(r.stato, StatoDecisione.IN_ATTESA)
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertTrue(record["decisione"]["honeypot"])
            self.assertEqual(record["extra"]["esito_tentato"], "APPROVATA")

    def test_honeypot_non_si_aggira_con_modifica_fittizia(self):
        with TemporaryDirectory() as directory:
            om = OversightManager(AuditLogger(str(Path(directory) / "audit.jsonl")))
            r = self._raccomandazione("A-HP-MOD", honeypot=True)
            om.coda.append(r)
            with self.assertRaises(ValueError):
                om.revisiona(r.id, StatoDecisione.MODIFICATA, "OP-01",
                             "Modifica apparentemente valida ma automatica",
                             azione_modificata=r.azione_proposta,
                             evidenze_keywords=["rischio", "confidenza"])
            self.assertEqual(r.stato, StatoDecisione.IN_ATTESA)

    def test_honeypot_corretto_con_azione_diversa_e_rilevato(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            om = OversightManager(AuditLogger(str(path)))
            r = self._raccomandazione("A-HP-CORRETTO", honeypot=True)
            r.livello = LivelloSupervisione.HITL
            om.coda.append(r)

            om.revisiona(r.id, StatoDecisione.MODIFICATA, "OP-01",
                         "Rischio basso incompatibile con intervento urgente",
                         azione_modificata="nessuna_azione",
                         evidenze_keywords=["rischio", "confidenza"])

            self.assertEqual(r.stato, StatoDecisione.MODIFICATA)
            metriche = metriche_rubber_stamping([
                json.loads(line) for line in path.read_text().splitlines()])
            self.assertEqual(metriche["honeypot_rilevati"], 1)
            self.assertEqual(metriche["honeypot_falliti"], 0)

    def test_raccomandazione_bloccata_da_stop_resta_in_coda(self):
        with TemporaryDirectory() as directory:
            om = OversightManager(AuditLogger(str(Path(directory) / "audit.jsonl")))
            om.attiva_stop("area:Nord", "OP-01", "Stop area per anomalia operativa confermata")
            r = self._raccomandazione("A-HP-STOP", honeypot=True)

            om.sottometti(r)

            self.assertIn(r, om.coda)
            self.assertEqual(r.stato, StatoDecisione.BLOCCATA_STOP)
            self.assertIsNotNone(r.livello)

    def test_metriche_rubber_stamping_separano_i_fenomeni(self):
        records = [
            {"attore": "OP-01", "evento": "revisione_APPROVATA",
             "decisione": {"id": "1", "motivazione": "testo breve"},
             "extra": {"tempo_esposizione_secondi": 4}},
            {"attore": "OP-01", "evento": "revisione_MODIFICATA",
             "decisione": {"id": "2", "motivazione": "testo breve"},
             "extra": {"tempo_esposizione_secondi": 14}},
            {"attore": "OP-02", "evento": "revisione_APPROVATA",
             "decisione": {"id": "3", "motivazione": "testo breve"},
             "extra": {}},
            {"attore": "OP-01", "evento": "allerta_honeypot_approvato",
             "decisione": {"id": "hp-1", "asset_id": "HP-A", "honeypot": True}},
            {"attore": "OP-01", "evento": "revisione_RIFIUTATA",
             "decisione": {"id": "hp-2", "asset_id": "HP-B", "honeypot": True}},
            {"attore": "SISTEMA", "evento": "in_coda_HITL",
             "decisione": {"id": "hp-3", "asset_id": "HP-C", "honeypot": True}},
        ]

        metriche = metriche_rubber_stamping(records)

        self.assertEqual(metriche["motivazioni_brevi"], 3)
        self.assertEqual(metriche["motivazioni_duplicate"], 1)
        self.assertEqual(metriche["approvazioni_rapide"], 1)
        self.assertEqual(metriche["revisioni_con_tempo"], 2)
        self.assertEqual(metriche["honeypot_esposti"], 3)
        self.assertEqual(metriche["honeypot_valutati"], 2)
        self.assertEqual(metriche["honeypot_falliti"], 1)
        self.assertEqual(metriche["honeypot_rilevati"], 1)

    def test_emergency_stop_multi_filtro_usa_and_tra_dimensioni(self):
        om = OversightManager(AuditLogger("test_audit_temp.jsonl"))
        nord = self._raccomandazione("A-5")
        sud_linea = self._raccomandazione("A-6")
        sud_linea.area_geografica = "Sud"
        sud_linea.tipo_asset = "linea_AT"
        centro_cabina = self._raccomandazione("A-7")
        centro_cabina.area_geografica = "Centro"
        centro_cabina.tipo_asset = "cabina_primaria"
        om.coda.extend([nord, sud_linea, centro_cabina])

        bloccate = om.attiva_stop_filtri(
            ["area:Sud", "tipo:linea_AT"], "OP-01",
            "Stop operativo su filtri multipli"
        )

        self.assertEqual(bloccate, 1)
        self.assertEqual(nord.stato, StatoDecisione.IN_ATTESA)
        self.assertEqual(sud_linea.stato, StatoDecisione.BLOCCATA_STOP)
        self.assertEqual(centro_cabina.stato, StatoDecisione.IN_ATTESA)
        self.assertEqual(om.stop_attivi, {"area:Sud", "tipo:linea_AT"})

    def test_riprendi_attivita_ripristina_decisioni_non_piu_in_stop(self):
        om = OversightManager(AuditLogger("test_audit_temp.jsonl"))
        nord = self._raccomandazione("A-8")
        sud_linea = self._raccomandazione("A-9")
        sud_linea.area_geografica = "Sud"
        sud_linea.tipo_asset = "linea_AT"
        om.coda.extend([nord, sud_linea])

        om.attiva_stop_filtri(["area:Nord"], "OP-01", "Stop operativo sulla sola area Nord")
        om.attiva_stop_filtri(["tipo:linea_AT"], "OP-01", "Stop operativo sulle sole linee alta tensione")
        ripristinate = om.disattiva_stop_filtri(["area:Nord"], "OP-01",
                                                "Ripresa attivita area Nord")

        self.assertEqual(ripristinate, 1)
        self.assertEqual(nord.stato, StatoDecisione.IN_ATTESA)
        self.assertEqual(sud_linea.stato, StatoDecisione.BLOCCATA_STOP)
        self.assertEqual(om.stop_attivi, {"tipo:linea_AT"})

    def test_emergency_stop_richiede_motivazione_descrittiva(self):
        om = OversightManager(AuditLogger("test_audit_temp.jsonl"))
        with self.assertRaises(ValueError):
            om.attiva_stop_filtri(["area:Nord"], "OP-01", "")
        with self.assertRaises(ValueError):
            om.attiva_stop_filtri(["area:Nord"], "OP-01", "troppo breve")
        with self.assertRaises(ValueError):
            om.attiva_stop_filtri(["area:Nord"], "OP-01", "Motivazione insufficiente")

        om.attiva_stop_filtri(["area:Nord"], "OP-01", "Guasto confermato su area Nord")
        self.assertEqual(om.stop_attivi, {"area:Nord"})

    def test_stop_globale_e_filtri_specifici_sono_esclusivi(self):
        with TemporaryDirectory() as directory:
            om = OversightManager(AuditLogger(str(Path(directory) / "audit.jsonl")))
            with self.assertRaises(ValueError):
                om.attiva_stop_filtri(["GLOBALE", "area:Nord"], "OP-01",
                                      "Tentativo combinazione globale non valida")
            om.attiva_stop("GLOBALE", "OP-01", "Arresto completo per anomalia critica")
            with self.assertRaises(ValueError):
                om.attiva_stop("area:Nord", "OP-01", "Arresto locale durante stop globale")

    def test_mitigazione_prudenziale_promuove_hotl_ed_e_auditata(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            om = OversightManager(AuditLogger(str(path)))
            r = self._raccomandazione("A-MIT")
            r.prob_guasto = 0.2
            r.azione_proposta = "ispezione_routine"

            self.assertEqual(om.route(r), LivelloSupervisione.HOTL)
            om.configura_mitigazione_prudenziale(True, ["drift oltre soglia"])
            self.assertEqual(om.route(r), LivelloSupervisione.HITL)
            om.configura_mitigazione_prudenziale(False, [], "OP-01")
            self.assertEqual(om.route(r), LivelloSupervisione.HOTL)

            eventi = [json.loads(riga)["evento"] for riga in path.read_text().splitlines()]
            self.assertEqual(eventi, ["mitigazione_prudenziale_ON", "mitigazione_prudenziale_OFF"])

    def test_audit_log_salva_messaggio_llm_nella_decisione(self):
        path = Path("test_audit_llm_temp.jsonl")
        if path.exists():
            path.unlink()
        audit = AuditLogger(str(path))
        r = self._raccomandazione("A-12")
        r.messaggio_llm = {
            "testo": "Spiegazione operativa",
            "incertezza": "Confidenza da verificare",
            "fonte": "llm:test/model",
            "latenza_ms": 42,
        }

        audit.log("OP-01", "revisione_APPROVATA", r)

        record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(record["decisione"]["messaggio_llm"]["testo"], "Spiegazione operativa")
        self.assertNotIn("llm", record["evento"].lower())
        path.unlink()

    def test_audit_con_due_istanze_mantiene_catena_integra(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit_a = AuditLogger(str(path))
            audit_b = AuditLogger(str(path))

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit((audit_a if indice % 2 else audit_b).log,
                                    "SISTEMA", f"evento_{indice}")
                    for indice in range(40)
                ]
                for future in futures:
                    future.result()

            self.assertEqual(AuditLogger(str(path)).verifica_catena(), (True, 40))

    def test_audit_archivia_catena_compromessa_senza_riscriverla(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = AuditLogger(str(path))
            audit.log("SISTEMA", "primo")
            with path.open("a", encoding="utf-8") as file:
                file.write('{"evento":"record_non_valido","hash":"errato"}\n')

            nuovo_audit = AuditLogger(str(path))

            self.assertEqual(nuovo_audit.verifica_catena(), (True, 1))
            migrazione = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(migrazione["evento"], "migrazione_catena_audit_compromessa")
            self.assertEqual(len(list(Path(directory).glob("audit.corrotto-*.jsonl"))), 1)

    def test_sla_scaduto_porta_hitl_in_escalation(self):
        om = OversightManager(AuditLogger("test_audit_temp.jsonl"), sla_minuti=30)
        r = self._raccomandazione("A-10")
        r.livello = LivelloSupervisione.HITL
        r.creata_il = (datetime.now() - timedelta(minutes=45)).isoformat()
        om.coda.append(r)

        escalation = om.aggiorna_sla()

        self.assertEqual(escalation, 1)
        self.assertEqual(r.stato, StatoDecisione.ESCALATION)

    def test_sla_non_scaduto_resta_in_attesa(self):
        om = OversightManager(AuditLogger("test_audit_temp.jsonl"), sla_minuti=30)
        r = self._raccomandazione("A-11")
        r.livello = LivelloSupervisione.HITL
        r.creata_il = (datetime.now() - timedelta(minutes=10)).isoformat()
        om.coda.append(r)

        escalation = om.aggiorna_sla()

        self.assertEqual(escalation, 0)
        self.assertEqual(r.stato, StatoDecisione.IN_ATTESA)


if __name__ == "__main__":
    unittest.main()
