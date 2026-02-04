def filter_activities(question: str, db: list):
    q = question.lower()

    # --- Status filter ---
    if "executed" in q:
        return [a for a in db if a.get("ActivityStatus") == "Executed"]

    if "planned" in q:
        return [a for a in db if a.get("ActivityStatus") == "Planned"]

    if "in progress" in q:
        return [a for a in db if a.get("ActivityStatus") == "In progress"]


    # --- Country filter ---
    for country in ["ghana", "nigeria", "kenya", "south africa"]:
        if country in q:
            return [
                a for a in db
                if any(
                    c.get("value", "").lower() == country
                    for c in a.get("CountriesSplitted", [])
                )
            ]

    # --- Beneficiary filter ---
    if "women" in q:
        return [
            a for a in db
            if "Women and Girls" in a.get("BeneficiariesExtracted", [])
        ]

    # --- Fallback: cap dataset ---
    return db[:50]   # safety cap
