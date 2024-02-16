OTHER = "Other"
TYPEWRITTEN = "Typewritten"

CLASSES = [OTHER, TYPEWRITTEN]

CLASS2INT = {c: i for i, c in enumerate(CLASSES)}
CLASS2NAME = {v: k for k, v in CLASS2INT.items()}
