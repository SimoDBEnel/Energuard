"""
EnerGuard Starter Kit - BiasDetector
=====================================
Metriche di performance e fairness disaggregate per sottogruppo.
Un modello "accurato in media" puo' nascondere disparita' gravi:
questo modulo serve a renderle visibili nella dashboard.

TODO per il team:
  1. Scegliere e giustificare le metriche di fairness rilevanti per QUESTO
     caso d'uso. Suggerimento: qui il danno peggiore e' il falso negativo
     (guasto non previsto su un asset critico), quindi il gap di recall /
     FNR tra aree pesa piu' della demographic parity.
  2. Definire soglie di allerta e collegarle a un alert visivo in dashboard.
  3. Indagare le anomalie: se due aree hanno profili di rischio simili ma
     tassi di guasto registrati molto diversi, cosa puo' significare per
     la qualita' delle label storiche? (vedi Art. 10, data governance)
"""

import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix)


class BiasDetector:
    def __init__(self, soglia_gap_recall: float = 0.15,
                 soglia_gap_selezione: float = 0.20):
        self.soglia_gap_recall = soglia_gap_recall
        self.soglia_gap_selezione = soglia_gap_selezione

    def metriche_per_gruppo(self, df: pd.DataFrame, col_gruppo: str,
                            col_y: str = "y_true",
                            col_pred: str = "y_pred") -> pd.DataFrame:
        righe = []
        for gruppo, sub in df.groupby(col_gruppo):
            tn, fp, fn, tp = confusion_matrix(
                sub[col_y], sub[col_pred], labels=[0, 1]).ravel()
            righe.append({
                col_gruppo: gruppo,
                "n": len(sub),
                "tasso_positivi_reali": round(sub[col_y].mean(), 3),
                "tasso_selezione": round(sub[col_pred].mean(), 3),  # per demographic parity
                "accuracy": round(accuracy_score(sub[col_y], sub[col_pred]), 3),
                "precision": round(precision_score(sub[col_y], sub[col_pred],
                                                   zero_division=0), 3),
                "recall": round(recall_score(sub[col_y], sub[col_pred],
                                             zero_division=0), 3),
                "fnr": round(fn / (fn + tp), 3) if (fn + tp) else None,
                "fpr": round(fp / (fp + tn), 3) if (fp + tn) else None,
            })
        return pd.DataFrame(righe).sort_values("recall")

    def allerte(self, metriche: pd.DataFrame, col_gruppo: str) -> list[str]:
        """Confronta ogni gruppo con il migliore e genera allerte testuali."""
        out = []
        m = metriche.dropna(subset=["recall"])
        if m.empty:
            return out
        best_recall = m["recall"].max()
        for _, r in m.iterrows():
            if best_recall - r["recall"] > self.soglia_gap_recall:
                out.append(
                    f"GAP RECALL: {col_gruppo}='{r[col_gruppo]}' ha recall "
                    f"{r['recall']} contro un massimo di {best_recall} "
                    f"(gap {round(best_recall - r['recall'], 3)} > soglia "
                    f"{self.soglia_gap_recall}). Rischio: guasti non previsti "
                    f"concentrati in questo sottogruppo.")
        gap_sel = m["tasso_selezione"].max() - m["tasso_selezione"].min()
        if gap_sel > self.soglia_gap_selezione:
            out.append(
                f"GAP SELEZIONE (demographic parity): differenza di "
                f"{round(gap_sel, 3)} nel tasso di interventi raccomandati tra "
                f"sottogruppi. Verificare se riflette rischio reale o bias "
                f"storico nei dati (variabile proxy).")
        return out

    def calibrazione_per_gruppo(self, df: pd.DataFrame, col_gruppo: str,
                               col_y: str = "y_true",
                               col_proba: str = "proba") -> pd.DataFrame:
        """Confronta probabilita' media prevista e frequenza osservata di guasto."""
        righe = []
        for gruppo, sub in df.groupby(col_gruppo):
            pred_media = float(sub[col_proba].mean()) if not sub.empty else 0.0
            osservato = float(sub[col_y].mean()) if not sub.empty else 0.0
            righe.append({
                col_gruppo: gruppo,
                "predizione_media": round(pred_media, 3),
                "osservato_media": round(osservato, 3),
                "differenza": round(osservato - pred_media, 3),
                "brier_score": round(((sub[col_proba] - sub[col_y]) ** 2).mean(), 3),
                "n": len(sub),
            })
        return pd.DataFrame(righe).sort_values("differenza", ascending=False)

    def drift_temporale(self, df: pd.DataFrame, col_gruppo: str,
                        col_y: str = "y_true", col_proba: str = "proba",
                        finestra: int = 20) -> pd.DataFrame:
        """Stima un drift sintetico in pseudo-tempo, utile quando non esiste un timestamp."""
        out = []
        for gruppo, sub in df.groupby(col_gruppo):
            sub = sub.sort_values(by=["asset_id"], kind="mergesort").reset_index(drop=True)
            for start in range(0, len(sub), finestra):
                blocco = sub.iloc[start:start + finestra]
                if len(blocco) < max(5, finestra // 2):
                    continue
                pred_media = float(blocco[col_proba].mean())
                osservato = float(blocco[col_y].mean())
                out.append({
                    col_gruppo: gruppo,
                    "window_start": start,
                    "window_end": start + len(blocco) - 1,
                    "predizione_media": round(pred_media, 3),
                    "osservato_media": round(osservato, 3),
                    "drift": round(osservato - pred_media, 3),
                    "n": len(blocco),
                })
        return pd.DataFrame(out)

    # TODO avanzato: equalized odds difference, rispetto ai gruppi protetti,
    # offset temporale reale con timestamp di produzione.
