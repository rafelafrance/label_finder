# #########################################################################
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

# #########################################################################
CLASSES = "Other Typewritten ".split()

CLASS2INT = {c: i for i, c in enumerate(CLASSES)}
CLASS2NAME = {v: k for k, v in CLASS2INT.items()}