"""Dynamic decision-support recommendation generator based on MHRI risk profile."""

from dataclasses import dataclass
from typing import Dict, List

from src.mcda_engine import MHRResult


@dataclass(frozen=True)
class ActionPlan:
    """Structured decision-support recommendations for citizens and municipalities."""

    summary_tr: str
    summary_en: str
    priority_hazards: List[str]
    citizen_actions: List[str]
    municipality_actions: List[str]


_HAZARD_LABELS_TR: Dict[str, str] = {
    "S_p": "Sismik Yakınlık (Fay Hattı / PGA)",
    "G_s": "Zemin & Sıvılaşma Potansiyeli",
    "T_e": "Topoğrafik Eğim & Heyelan Riski",
    "H_m": "Hidro-Meteorolojik Taşkın Yükü",
    "E_d": "Yapısal Maruziyet & Bina Stoğu",
}

_SUMMARIES: Dict[str, Dict[str, str]] = {
    "Low Risk": {
        "tr": "Bölge düşük çoklu afet riski taşımaktadır. Rutin hazırlık ve periyodik kontroller yeterlidir.",
        "en": "The location exhibits low multi-hazard risk. Routine preparedness and periodic checks are sufficient.",
    },
    "Moderate Risk": {
        "tr": "Orta düzeyde çoklu afet riski mevcuttur. Öne çıkan tehlike parametreleri için önleyici tedbirler planlanmalıdır.",
        "en": "Moderate multi-hazard risk detected. Preventive measures should be planned for elevated hazard drivers.",
    },
    "High Risk": {
        "tr": "Yüksek risk seviyesi tespit edildi. Yapısal güçlendirme, zemin iyileştirme ve aktif erken uyarı önlemleri önceliklendirilmelidir.",
        "en": "High risk level detected. Structural retrofitting, ground improvement, and active early-warning measures must be prioritized.",
    },
    "Critical Risk": {
        "tr": "KRİTİK RİSK: Çoklu afet tetiklenme potansiyeli yüksektir. Acil müdahale planları, kentsel dönüşüm ve tahliye koridorları derhal devreye alınmalıdır.",
        "en": "CRITICAL RISK: High multi-hazard cascading potential. Emergency response plans, urban transformation, and evacuation corridors must be activated immediately.",
    },
}


def generate_recommendations(mcda: MHRResult) -> ActionPlan:
    """Generate tailored citizen and local government action plans from an MHRResult."""
    params = mcda.parameters
    priority_hazards = [
        _HAZARD_LABELS_TR[k] for k, v in sorted(params.items(), key=lambda item: item[1], reverse=True) if v >= 0.50
    ]
    if not priority_hazards:
        top_key = max(params, key=lambda k: params[k])
        priority_hazards = [_HAZARD_LABELS_TR[top_key]]

    citizen_actions: List[str] = []
    municipality_actions: List[str] = []

    if mcda.cascading_triggered:
        triggered_names = ", ".join(_HAZARD_LABELS_TR[k] for k in mcda.triggered_parameters)
        citizen_actions.append(
            f"Zincirleme afet uyarısı (α={mcda.alpha}): Birden fazla kritik eşik aşıldı ({triggered_names}). "
            "İkincil afetlere (heyelan, sıvılaşma, taşkın) karşı alternatif tahliye rotası belirleyin."
        )
        municipality_actions.append(
            f"Zincirleme etkileşim çarpanı aktif (α={mcda.alpha}): Bütünleşik Çoklu Afet Acil Eylem Planı'nı "
            "devreye alın ve kritik altyapı (enerji, su, ulaşım) yedekliliğini denetleyin."
        )

    if params["S_p"] >= 0.50:
        citizen_actions.append(
            "72 saatlik afet ve acil durum çantası hazırlayın; yaşam üçgeni oluşturabilecek güvenli noktaları belirleyin ve eşyaları sabitleyin."
        )
        municipality_actions.append(
            "Aktif fay zonuna yakın parsellerde fay sakınım bandı (tampon bölge) kısıtlamalarını uygulayın ve deprem erken uyarı sensör ağını sıklaştırın."
        )

    if params["G_s"] >= 0.50:
        citizen_actions.append(
            "Binanızın zemin etüdü ve temel tipi (radye/kazıklı temel) bilgilerini e-Devlet/belediye kayıtlarından doğrulayın."
        )
        municipality_actions.append(
            "Yüksek sıvılaşma potansiyelli alüvyon zeminlerde mikro-bölgeleme etütlerini güncelleyin ve jet-grout/kazıklı temel zorunluluğu getirin."
        )

    if params["T_e"] >= 0.50:
        citizen_actions.append(
            "Dik yamaç altlarında istinat duvarı çatlaklarını ve zemin kabarmalarını düzenli gözlemleyin; şiddetli yağışta alt katlardan uzak durun."
        )
        municipality_actions.append(
            "Eğimli bölgelerde şev stabilitesi analizleri yapın, istinat yapılarını güçlendirin ve yamaç drenaj kanalları inşa edin."
        )

    if params["H_m"] >= 0.50:
        citizen_actions.append(
            "Bodrum ve zemin katlarda geri tepme vanası (çekvalf) kullanın; meteorolojik turuncu/kırmızı kod uyarılarında dere yataklarından uzak durun."
        )
        municipality_actions.append(
            "Yağmur suyu drenaj kapasitesini artırın, dere yataklarındaki yapılaşmayı kaldırın ve taşkın erken uyarı istasyonları kurun."
        )

    if params["E_d"] >= 0.50:
        citizen_actions.append(
            "2000 yılı öncesi yapılmış binalarda hızlı tarama veya performans analizi yaptırın; DASK poliçenizi güncel tutun."
        )
        municipality_actions.append(
            "Yaşlı ve yoğun yapı stoğu bulunan mahallelerde öncelikli kentsel dönüşüm ve yapısal güçlendirme teşvik programları başlatın."
        )

    if not citizen_actions:
        citizen_actions.append(
            "Temel aile afet planı oluşturun, en yakın AFAD toplanma alanını öğrenin ve yılda bir kez tatbikat yapın."
        )
    if not municipality_actions:
        municipality_actions.append(
            "Mevcut imar planlarında afet risk katmanlarını güncel tutun ve periyodik altyapı bakımını sürdürün."
        )

    summary = _SUMMARIES.get(mcda.risk_level_en, _SUMMARIES["Moderate Risk"])
    return ActionPlan(
        summary_tr=summary["tr"],
        summary_en=summary["en"],
        priority_hazards=priority_hazards,
        citizen_actions=citizen_actions,
        municipality_actions=municipality_actions,
    )
