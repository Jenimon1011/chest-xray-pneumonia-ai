# Chest X-ray Pneumonia Classifier with Grad-CAM

Deep learning model that classifies pediatric chest X-rays as **Normal** or **Pneumonia**, explains each prediction with **Grad-CAM** heatmaps, and runs as an interactive **Streamlit web app**.

> ⚠️ Research and education project only. Not a medical device and not for clinical use.

<!-- After training, replace these with your own images from the results/ folder -->
![Grad-CAM examples](results/gradcam_examples.png)

## Highlights

- **Transfer learning:** EfficientNetB0 pretrained on ImageNet, fine-tuned in two phases
- **Class imbalance handling:** class weights (about 3× more pneumonia than normal images)
- **Clinically relevant metrics:** sensitivity, specificity, ROC AUC, not just accuracy
- **Explainability:** Grad-CAM shows which lung regions drove each prediction
- **Deployment:** Streamlit app for uploading an image and getting a prediction plus heatmap
## Results (held-out test set, n = 624)

| Metric | Value |
|---|---|
| Accuracy | 92.1% |
| Sensitivity (pneumonia recall) | 97.2% |
| Specificity (normal recall) | 83.8% |
| Precision (Normal / Pneumonia) | 0.947 / 0.909 |
| F1 (Normal / Pneumonia) | 0.889 / 0.939 |
| ROC AUC | 0.974 |

|  | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | 196 | 38 |
| **Actual Pneumonia** | 11 | 379 |

The model favours sensitivity: it misses 11 of 390 pneumonia cases, but 38 of 234
healthy X-rays are flagged as pneumonia.

**Limitations:** paediatric images from a single centre, no external validation.
This is a research prototype, not a medical device.


## Dataset

[PneumoniaMNIST](https://medmnist.com/) (MedMNIST v2/v3, 224×224 version), derived from the pediatric chest X-ray dataset of Kermany et al., *Cell* 2018. Official train/val/test split: 4,708 / 524 / 624 images. License: CC BY 4.0.

## Project structure

```
chest-xray-pneumonia-ai/
├── notebooks/
│   └── 01_train_pneumonia_classifier.ipynb   # data → training → evaluation → Grad-CAM
├── models/                                   # trained model goes here (pneumonia_model.keras)
├── results/                                  # plots for this README
├── sample_images/                            # test X-rays to try in the app
├── app.py                                    # Streamlit web app
├── gradcam.py                                # Grad-CAM helper functions
└── requirements.txt
```

## How to run

**1. Train (Google Colab, free GPU):** open `notebooks/01_train_pneumonia_classifier.ipynb` in Colab, select a T4 GPU, run all cells, and download `project_outputs.zip`.

**2. Run the app locally:**

```bash
git clone https://github.com/<your-username>/chest-xray-pneumonia-ai.git
cd chest-xray-pneumonia-ai
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
# put pneumonia_model.keras into models/
streamlit run app.py
```

## Limitations

- Pediatric images from a single hospital, so performance on adult or other-hospital X-rays is unknown (domain shift).
- Binary task only; it can't tell bacterial from viral pneumonia or detect other findings.
- Grad-CAM is a coarse explanation and doesn't prove the model "understands" pathology.
- If this were developed as a real product, it would be software as a medical device under EU MDR, with risk management (ISO 14971) and clinical evaluation required.

## Next steps

- [ ] Test on an external dataset (e.g. RSNA Pneumonia) to measure generalisation
- [ ] Tune the decision threshold for higher sensitivity
- [ ] Deploy publicly on Streamlit Community Cloud

## Author

**Jenimon Hariprasad**, M.Sc. Biomedical Engineering, Hochschule Anhalt
[LinkedIn](https://linkedin.com/in/jenimon-hariprasad-1ab964198) · [Portfolio](https://jenimon1011.github.io/jenimon-hariprasad/)

---

# Beginner walkthrough (delete this section before publishing)

Work through it in order. Tick each step off as you go.

### Week 1: set up and train

1. **Create accounts:** [GitHub](https://github.com) (if not done) and sign in to [Google Colab](https://colab.research.google.com) with your Google account.
2. **Create the repo:** on GitHub click **New repository**, name it `chest-xray-pneumonia-ai`, make it **Public**, and tick "Add a README". Then use **Add file → Upload files** to upload everything from this folder (keep the `notebooks/` folder).
3. **Open the notebook in Colab:** in Colab go to **File → Open notebook → GitHub**, paste your repo URL, and pick the notebook.
4. **Turn on the GPU:** **Runtime → Change runtime type → T4 GPU**.
5. **Run each cell one by one** (`Shift + Enter`). Read the text above each cell before running it. Training takes about 10–20 minutes.
6. **Write down your numbers** from section 7 (sensitivity, specificity, AUC).
7. The last cell downloads `project_outputs.zip`. Unzip it on your laptop.
8. **Save the notebook with outputs back to GitHub:** **File → Save a copy in GitHub**. The plots will then show on GitHub.

### Week 2: run the app on your laptop

1. Install [Python 3.11](https://www.python.org/downloads/) (tick "Add Python to PATH") and [VS Code](https://code.visualstudio.com/).
2. Install [Git](https://git-scm.com/), then in a terminal: `git clone https://github.com/<your-username>/chest-xray-pneumonia-ai.git`
3. Copy `models/pneumonia_model.keras`, `results/` and `sample_images/` from the zip into the project folder.
4. Run the commands from **How to run → step 2** above. A browser tab opens. Upload an image from `sample_images/`.
5. Take a **screenshot of the app** and save it as `results/app_screenshot.png`.

### Week 3: polish and publish

1. Fill in the results table and add the screenshot to this README.
2. Push to GitHub:
   ```bash
   git add .
   git commit -m "Add trained model, results and app"
   git push
   ```
   (`pneumonia_model.keras` is about 20 MB, which is fine for GitHub.)
3. **Optional, but impressive:** deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud). Choose "New app", select your repo, and set the main file to `app.py`. You get a public link you can add to your CV.
4. Add the project to your portfolio website and LinkedIn (**Featured → Add a link**).
5. Delete this walkthrough section.

### Words you'll meet

| Term | Meaning |
|---|---|
| Transfer learning | Reuse a network trained on millions of everyday photos, then adapt it to X-rays |
| Epoch | One full pass through all training images |
| Validation set | Images used to choose the best model during training |
| Test set | Images kept aside until the very end for an honest score |
| Augmentation | Small random changes (rotate, zoom) so the model doesn't memorise |
| Class weights | Make the rarer class count more in the loss |
| Sensitivity | Share of real pneumonia cases the model catches |
| Specificity | Share of healthy cases correctly called normal |
| Grad-CAM | Heatmap of which regions influenced the prediction |
