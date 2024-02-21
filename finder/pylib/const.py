OTHER: str = "Other"
TYPEWRITTEN: str = "Typewritten"

CLASSES: list[str] = [OTHER, TYPEWRITTEN]

CLASS2INT: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
CLASS2NAME: dict[int, str] = {v: k for k, v in CLASS2INT.items()}
