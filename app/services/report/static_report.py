from app.schemas.response import PredictionStatus

# Disease knowledge base 
# Each entry: { "summary": str, "recommended_treatment": str }
DISEASE_REPORTS: dict[str, dict] = {
    "Tomato_Bacterial_Spot": {
        "summary": (
            "Bacterial spot is caused by Xanthomonas spp. and appears as small, "
            "water-soaked lesions that turn dark brown with yellow halos. "
            "It spreads rapidly in warm, wet conditions and can affect leaves, stems, and fruit."
        ),
        "recommended_treatment": (
            "1. Remove and destroy infected plant material immediately.\n"
            "2. Apply copper-based bactericides (e.g. copper hydroxide) every 7–10 days.\n"
            "3. Avoid overhead irrigation — use drip irrigation to keep foliage dry.\n"
            "4. Rotate crops and use certified disease-free seeds next season."
        ),
    },
    "Tomato_Early_Blight": {
        "summary": (
            "Early blight is caused by the fungus Alternaria solani. "
            "It produces characteristic dark brown lesions with concentric rings (target-board pattern) "
            "on older leaves first, gradually moving up the plant."
        ),
        "recommended_treatment": (
            "1. Remove affected lower leaves to slow spread.\n"
            "2. Apply fungicides containing chlorothalonil or mancozeb every 7–14 days.\n"
            "3. Mulch around plants to prevent soil splash.\n"
            "4. Ensure adequate plant spacing for airflow.\n"
            "5. Water at the base of plants, not overhead."
        ),
    },
    "Tomato_Healthy": {
        "summary": (
            "The leaf appears healthy with no visible signs of disease, pest damage, "
            "or nutritional deficiency. The plant is in good condition."
        ),
        "recommended_treatment": (
            "No treatment required. Continue regular care:\n"
            "1. Water consistently and avoid waterlogging.\n"
            "2. Monitor weekly for early signs of disease or pests.\n"
            "3. Apply balanced fertilizer as needed."
        ),
    },
    "Tomato_Late_Blight": {
        "summary": (
            "Late blight is caused by Phytophthora infestans — the same pathogen behind "
            "the Irish Potato Famine. It creates large, greasy-looking dark lesions on leaves "
            "and can destroy an entire crop within days under cool, moist conditions."
        ),
        "recommended_treatment": (
            "1. Act immediately — late blight spreads extremely fast.\n"
            "2. Remove and bag (do not compost) all infected material.\n"
            "3. Apply systemic fungicides (e.g. metalaxyl, cymoxanil) promptly.\n"
            "4. Avoid working in the field when plants are wet.\n"
            "5. Destroy severely infected plants to protect neighbouring crops."
        ),
    },
    "Tomato_Leaf_Mold": {
        "summary": (
            "Leaf mold is caused by Passalora fulva (formerly Fulvia fulva). "
            "It appears as pale green or yellow patches on the upper leaf surface "
            "with olive-green to grey mold growth on the underside. "
            "It thrives in high humidity and poor ventilation."
        ),
        "recommended_treatment": (
            "1. Improve greenhouse/tunnel ventilation to reduce humidity below 85%.\n"
            "2. Apply fungicides containing chlorothalonil or mancozeb.\n"
            "3. Remove and destroy heavily infected leaves.\n"
            "4. Avoid wetting leaves during irrigation.\n"
            "5. Consider resistant varieties for future planting."
        ),
    },
    "Tomato_Mosaic_Virus": {
        "summary": (
            "Tomato mosaic virus (ToMV) causes mottled light and dark green patterns on leaves, "
            "leaf distortion, and stunted growth. It is highly contagious and spreads through "
            "contact, contaminated tools, and infected seeds. There is no cure once a plant is infected."
        ),
        "recommended_treatment": (
            "1. There is no chemical cure — management is purely preventive.\n"
            "2. Remove and destroy infected plants immediately to protect others.\n"
            "3. Disinfect all tools with 10% bleach solution or 70% alcohol.\n"
            "4. Wash hands thoroughly before handling healthy plants.\n"
            "5. Use virus-free certified seed and resistant varieties.\n"
            "6. Control aphid populations as they can carry the virus."
        ),
    },
    "Tomato_Septoria_Leaf_Spot": {
        "summary": (
            "Septoria leaf spot is caused by the fungus Septoria lycopersici. "
            "It presents as numerous small, circular spots with dark brown borders and "
            "lighter centres, often with tiny black dots (pycnidia) visible in the centre. "
            "It typically starts on lower leaves after fruiting begins."
        ),
        "recommended_treatment": (
            "1. Remove and destroy infected lower leaves promptly.\n"
            "2. Apply fungicides with chlorothalonil, mancozeb, or copper every 7–10 days.\n"
            "3. Avoid overhead watering and working among wet plants.\n"
            "4. Mulch soil to reduce splash-spread of spores.\n"
            "5. Practice 2–3 year crop rotation."
        ),
    },
    "Tomato_Yellow_Leaf_Curl": {
        "summary": (
            "Tomato yellow leaf curl virus (TYLCV) is transmitted by the silverleaf whitefly "
            "(Bemisia tabaci). Infected plants show severe upward leaf curling, yellowing, "
            "and stunted growth. Young plants are most vulnerable and yield loss can reach 100%."
        ),
        "recommended_treatment": (
            "1. Remove and destroy infected plants immediately.\n"
            "2. Apply insecticides targeting whiteflies (e.g. imidacloprid, spiromesifen).\n"
            "3. Use yellow sticky traps to monitor and reduce whitefly populations.\n"
            "4. Install fine mesh screens in greenhouses.\n"
            "5. Plant resistant/tolerant tomato varieties.\n"
            "6. Avoid planting near other infected crops."
        ),
    },
}


def generate_report(
    predicted_class: str | None,
    status: PredictionStatus,
) -> tuple[str | None, str | None]:
    """
    Returns (summary, recommended_treatment) for a given prediction.

    - FAILED status → returns (None, None) — no report for invalid images
    - Unknown class → returns a generic fallback message
    """
    if status == PredictionStatus.FAILED or predicted_class is None:
        return None, None

    entry = DISEASE_REPORTS.get(predicted_class)
    if not entry:
        return (
            f"Disease class '{predicted_class}' was detected but no detailed report is available yet.",
            "Please consult an agricultural specialist for treatment advice.",
        )

    return entry["summary"], entry["recommended_treatment"]