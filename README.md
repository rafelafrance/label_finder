# label_finder ![Python application](https://github.com/rafelafrance/label_finder/workflows/CI/badge.svg)

Use a neural network to find labels on a herbarium sheets.

1. [Description](#Description)
2. [Install](#Install)
3. [Scenario: Label inference](#Label-inference)
4. [Scenario: Model training](#Model-training)

## Description

We find labels with a custom trained YOLOv7 model (https://github.com/WongKinYiu/yolov7). You will need to download and setup this repository separately, and run its scripts for inference and training.

[<img src="assets/show_labels.png" width="500" />](assets/show_labels.png)

- Labels that the model classified as typewritten are outlined in orange
- Other identified labels are outlined in teal.
- All label are extracted into their own image file, which may be used as input for an OCR engine.

## Install

You will need Git to clone this repository. You will also need to have Python3.11+ installed, as well as pip, a package manager for Python.
You can install the requirements into your python environment like so:

```bash
git clone https://github.com/rafelafrance/label_finder.git
cd /path/to/label_finder
make install
```

You will also need to install [YOLO7](https://github.com/WongKinYiu/yolov7). You may follow the instructions given there. I find that if you clone the repository, set a virtual environment for that repository, and install the requirements into that virtual environment things work just fine.

Every time you want to run any scripts in a new terminal session you will need to activate the virtual environment, once, before running them.

```bash
cd /path/to/label_finder
source .venv/bin/activate
```

## Label inference

### Requirements

1. You must set up this and the YOLO7 repositories.
2. You will need a YOLO7 model trained to find labels on herbarium sheets. I have an example one in this [zenodo]() link.
3. You also need some images of herbarium sheets. I have some sample images in the same [zenodo]() link.

### Optional: Clean up file names

I had a problem where herbarium sheet file names were given as URLs, and it confused some modules, so I renamed the files to remove problem characters.

#### Example

```bash
fix-herbarium-sheet-names --sheet-dir ./data/herbarium/sheets
```

### Prepare the images for YOLO

The images of herbarium sheets come in all different sizes. The model is trained on square images of a fixed size. The demo model was trained on 640x640 pixel color images. You need to resize the images to be square.

#### Example

```bash
yolo-inference --sheet-csv ./data/herbarium/sheets --yolo-images ./data/yolo/inference --yolo-size 640
```

### Run the YOLO model

After training, I move the best trained model into `./data/yolo/models`. I only mention this because I need this as an argument below.

**Note that you are running this script from the virtual environment in the yolo directory.**

This is an example of how to run inference.

```bash
python detect.py \
--weights ../label_finder/data/yolo/models/best.pt \
--source ../label_finder/data/yolo/inference \
--project ../label_finder/data/yolo/runs/inference/ \
--name run_2024-02-14a \
--exist-ok \
--nosave \
--save-txt \
--save-conf \
--conf-thres 0.1
```

- device: (Option not shown) If you want to use the CPU for inference then add the `--device cpu` option
- weights: Model path
- source: Directory containing images formatted for YOLO
- save-txt: Save the results to the text files (one per input image)
- save-conf: Save confidences in the result files
- project: The root directory for saving the results
- name: Save results to this directory under the project directory
- exist-ok: It's ok for the output directory to exist
- nosave: Do not save images
- conf-thres: Confidence threshold before saving the label

### Create labels from YOLO results

After we've run YOLO, we need to take the results and put them back into a format we can use. Mostly, we're cutting the label images out of the herbarium sheet images. There is also image scaling and other things going on here.

#### Example

```bash
yolo-results-to-labels --yolo-labels ./data/yolo/runs/inference/run_2024-02-14a/labels --sheet-dir ./data/herbarium/sheets --label-dir ./data/herbarium/labels
```

Note that the `labels` is always created under the `name` dir, and "labels" refers to the labels given by the YOLO model and not herbarium labels. The names for the label images have this format:
`<sheet stem>_<label class>_<left>_<top>_<right>_<bottom>.<sheet suffix>`

If the sheet is named: `248106.jpg`, then a label may be named `248106_Typewritten_1261_51_1646_273.jpg`.

### Optional: Filter typewritten labels

This moves all labels that are classified as "Typewritten" into a separate directory. The OCR works best on  typewritten labels or barcodes with printing. It will do a fair job when labels with handwriting or a mix of typewriting & handwriting are neatly printed.

I have noticed that the current example YOLO model tends to have a fair number of false positives but close to zero false negatives. Manually pruning the false positives is much easier than sorting all labels. YMMV.

#### Example

`get-typewritten-labels --label-dir ./data/herbarium/labels --typewritten-dir ./data/herbarium/typewritten`

## Model training

To train a supervised model like YOLO you need data, and preferably lots of it. Which is a time-consuming task. We could have done this ourselves -- and maybe we should have -- but we opted to crowdsource this. To do this we used Notes from Nature, which is part of the [Zooniverse](https://www.zooniverse.org/), a scientifically oriented platform that crowdsources gathering biological data. The individual data gathering projects are called "expeditions".

The expedition we created for this seemed straight-forward; give the volunteers an image of a herbarium sheet like the one shown above and have the volunteers draw the orange and teal boxes on the sheets. We have 3 volunteers draw the boxes on the same sheet, and I would reconcile the box coordinates for each sheet.

### Build expedition

We package up a bunch of herbarium sheet images in Zooniverse format. Experts built the actual expedition code and workflow.

#### Example

```bash
build-expedition --sheet-dir ./data/herbarium/sheets --expedition-dir ./data/expeditions/expedition_01 --reduce-by 2
```

### Reconcile expedition

#### Notes

Oh, the best laid plans...

The data we got back from the expeditions were not always of the highest quality. Most people did excellent work but a significant portion either did not "get" the box drawing task or were actively subverting the data collection process. Which means that it wasn't just a matter of finding overlapping boxes and taking the average of the box coordinate. Some folks drew a single box enclosing several labels (TODO show this), other times the boxes really didn't really cover the entire label, or had excessive borders around the label, and yet others just drew boxes randomly.

If you can find 2 (or more) of 3 people that agree on a box then you can use that. Agreement here is defined as the overlap area is significant, measured as the box's intersection over union (IoU). For example an IoU >= 0.85. We also want the box categories (typewritten, etc.) to match. We wound up throwing away a lot of data.

#### Prepare input

You will need the output data from the Zooniverse expedition and you will also need access to the [label reconciliation script](https://github.com/juliema/label_reconciliations)

Convert the output from Zooniverse into an "unreconciled" CSV like this:

```bash
cd /path/to/label_reconciliations
source .venv/bin/activate
./reconcile.py --unreconciled-csv ./data/expeditions/expedition_01.unreconciled.csv ./data/expeditions/expedition_01.csv
```

#### Example

```bash
reconcile-expedition --unreconciled-csv ./data/expeditions/expedition_01.unreconciled.csv --reconciled-csv ./data/expeditions/expedition_01.reconciled.csv --expand-by 2
```

**Note that the --expand-by factor must match the --reduce-by factor.**

### Train model

TODO

```bash
python train.py \
--weights yolov7.pt \
--data data/custom.yaml \
--workers 4 \
--batch-size 4 \
--cfg cfg/training/yolov7.yaml \
--name yolov7 \
--hyp data/hyp.scratch.p5.yaml \
--epochs 100 \
--exist-ok
```
