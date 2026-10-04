# Trash2Taste 🍽️♻️

**Turning cafeteria trash into smarter menus.**

Trash2Taste is an AI-powered food-waste reduction system for school cafeterias, built at **HackByte**. A camera mounted inside a trash can identifies the food being thrown away, that waste data is logged and enriched, and a machine-learning model uses it to recommend weekly menus that students are less likely to waste.

---

## How it works

```
 ┌──────────────┐    CSV log     ┌──────────────┐   enriched rows   ┌──────────────────┐
 │  Trash-can   │ ─────────────▶ │  Conversion  │ ────────────────▶ │  Waste dataset   │
 │  camera app  │  (Flask, :5001)│  + GPT tags  │                   │  (.csv)          │
 └──────────────┘                └──────────────┘                   └────────┬─────────┘
   YOLOv8 detect +                                                           │ retrain
   food classifier                                                           ▼
                                 ┌──────────────┐   predicted waste  ┌──────────────────┐
                                 │  Weekly menu │ ◀───────────────── │ Random Forest    │
                                 │  (GPT-4 +    │                    │ waste predictor  │
                                 │  ranking)    │                    └──────────────────┘
                                 └──────────────┘
```

1. **Detect waste.** A Kivy camera app (`Appbasic.py`) runs two models on every frame:
   - **YOLOv8n** (COCO) for general object detection, filtered to food classes
   - A **custom YOLOv8 classifier** fine-tuned on the Food-101 / Food-41 dataset (94 dishes)

   Results are smoothed across the last 5 frames by majority vote. Anything with average confidence ≥ 30% is logged to `combined_detection_log.csv`.
2. **Collect the log.** `Server_Cam.py` serves the log over HTTP, and `client.py` / `fetch.py` pull it to the main machine.
3. **Enrich the data.** `conversion.py` removes duplicate detections (5-second window per item), asks OpenAI for each item's attributes (category, meat/dairy/veg, fried, sweet, spicy, serving size…), and appends the rows to the waste dataset.
4. **Retrain.** `transfer_learning.py` updates the label encoders and retrains the **Random Forest** waste-prediction pipeline on the combined dataset.
5. **Generate menus.** `menu_generation.py` asks GPT-4 for 20 candidate cafeteria meals, scores each one with the waste model over repeated random weekly samples, and returns the 5 meals with the lowest predicted waste.
6. **Web front end.** `about.html`, `menu.html` and `contact.html` (+ `styles.css`) form the Trash2Taste site, including a weekly menu generator and a waste breakdown view.

---

## Tech stack

| Area | Tools |
|---|---|
| Computer vision | Ultralytics YOLOv8, PyTorch / TorchScript, torchvision, OpenCV |
| Camera app | Kivy |
| ML / data | scikit-learn (Random Forest), pandas, NumPy, joblib, Faker |
| LLM | OpenAI API (GPT-3.5-turbo, GPT-4) |
| Networking | Flask, requests |
| Front end | HTML, CSS, JavaScript |

---

## Repository layout

```
├── Appbasic.py                  # Kivy camera app: YOLO detection + custom food classifier, logs waste
├── Server_Cam.py                # Flask server exposing the detection log (port 5001)
├── client.py / fetch.py         # Pull the detection log from the camera device
├── conversion.py                # Dedupe detections, tag attributes via OpenAI, append to dataset
├── transfer_learning.py         # Retrain the waste-prediction model on new data
├── model.py                     # Same retraining logic wrapped in a function
├── menu_generation.py / main.py # GPT-4 menu candidates ranked by predicted waste
├── synthetic_data_generation.py # Generates the synthetic cafeteria waste dataset (~200 school days)
├── Tuning_YOLO.py               # Train/val split + YOLOv8 classification fine-tuning (Colab)
├── YOLO_Convertor.py            # Export the tuned classifier to TorchScript
├── OG_YOLO_Conversion.py        # Export base YOLO to TorchScript
├── *.html, styles.css           # Trash2Taste website
├── *.pt                         # YOLO weights (base, tuned, TorchScript)
├── *.pkl                        # Trained waste model + label encoders
└── *.csv                        # Synthetic, collected and correlated waste data
```

---

## Getting started

### 1. Install dependencies

```bash
pip install ultralytics torch torchvision opencv-python kivy pillow \
            scikit-learn pandas numpy joblib faker flask requests "openai<1.0"
```

> The code uses the legacy `openai.ChatCompletion` API, so install `openai<1.0`.

### 2. Configure

- **OpenAI key:** set `openai.api_key` in `conversion.py` and `menu_generation.py`. A better option is to read it from an environment variable: `openai.api_key = os.environ["OPENAI_API_KEY"]`. **Never commit your key.**
- **Model paths:** `Appbasic.py`, `Server_Cam.py`, `YOLO_Convertor.py` and `OG_YOLO_Conversion.py` contain absolute Windows paths. Change them to point to your local `Custom.pt` / `YOLO_Model.pt` / log file.
- **Device IP:** update `YOLO_IP` in `client.py` (and the URL in `fetch.py`) to the camera machine's IP address.

### 3. Run the pipeline

```bash
# On the camera device
python Appbasic.py          # start detecting and logging waste
python Server_Cam.py        # serve the log on :5001

# On the main machine
python fetch.py             # download the log
python -c "import conversion; conversion.process_detections('downloaded_file.csv')"
python transfer_learning.py # retrain the waste model
python menu_generation.py   # print the optimized weekly menu
```

Open `about.html` in a browser to view the website.

---

## Limitations & future work

- The waste model was first trained on **synthetic data** (`synthetic_data_generation.py`). Real predictions improve only as real detections are collected.
- Detection counts *items* thrown away, not the *weight* wasted. Adding a scale or volume estimation would make the data more accurate.
- Menu features currently use default attribute values. Using GPT attribute tagging for menu candidates too would sharpen the predictions.
- Nutritional compliance checks are a placeholder.

---

## Team

Built at HackByte by **Ahan Bhowmik** and **Advay**.
