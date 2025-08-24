# Complete Setup Guide for Credit Card Fraud Detection Project

## Step-by-Step Instructions

### Step 1: Download the Project Files
Download these 4 files from this project:
- `fraud_detection_notebook.ipynb` - The Jupyter notebook
- `fraud_detection.py` - Python script version
- `requirements.txt` - Required packages
- `README.md` - Project documentation

**How to download:**
1. Click on each file link in the interface
2. Right-click and select "Save as" or click the download button
3. Save all files to your Downloads folder first

### Step 2: Create Project Folder Structure
Open Terminal (Mac/Linux) or Command Prompt (Windows):

```bash
# Navigate to your desired location (e.g., Desktop)
cd ~/Desktop

# Create main project folder
mkdir fraud_detection

# Navigate into the folder
cd fraud_detection

# Create required subfolders
mkdir data
mkdir models
mkdir figures
```

**Windows users:** Replace `~/Desktop` with `C:\Users\YourUsername\Desktop`

### Step 3: Move Downloaded Files
Move the 4 downloaded files into the `fraud_detection` folder:

**Option A - Using Terminal/Command:**
```bash
# Mac/Linux
mv ~/Downloads/fraud_detection_notebook.ipynb ~/Desktop/fraud_detection/
mv ~/Downloads/fraud_detection.py ~/Desktop/fraud_detection/
mv ~/Downloads/requirements.txt ~/Desktop/fraud_detection/
mv ~/Downloads/README.md ~/Desktop/fraud_detection/

# Windows
move C:\Users\YourUsername\Downloads\fraud_detection_notebook.ipynb C:\Users\YourUsername\Desktop\fraud_detection\
move C:\Users\YourUsername\Downloads\fraud_detection.py C:\Users\YourUsername\Desktop\fraud_detection\
move C:\Users\YourUsername\Downloads\requirements.txt C:\Users\YourUsername\Desktop\fraud_detection\
move C:\Users\YourUsername\Downloads\README.md C:\Users\YourUsername\Desktop\fraud_detection\
```

**Option B - Manual:** 
Drag and drop files from Downloads to the fraud_detection folder

### Step 4: Verify Folder Structure
Your folder should look like this:
```
fraud_detection/
│
├── data/                    # Empty folder (will add dataset here)
├── models/                  # Empty folder (models will be saved here)
├── figures/                 # Empty folder (plots will be saved here)
├── fraud_detection_notebook.ipynb
├── fraud_detection.py
├── requirements.txt
└── README.md
```

### Step 5: Download the Dataset from Kaggle

1. **Create Kaggle Account** (if you don't have one):
   - Go to https://www.kaggle.com
   - Click "Register" and create a free account

2. **Download the Dataset:**
   - Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Click the "Download" button (you must be logged in)
   - This will download `creditcard.csv` (about 144 MB)

3. **Move Dataset to Project:**
   ```bash
   # Move the CSV file to the data folder
   mv ~/Downloads/creditcard.csv ~/Desktop/fraud_detection/data/
   
   # Or for Windows:
   move C:\Users\YourUsername\Downloads\creditcard.csv C:\Users\YourUsername\Desktop\fraud_detection\data\
   ```

### Step 6: Set Up Python Environment

**Option A - Using venv (Recommended):**
```bash
# Navigate to project folder
cd ~/Desktop/fraud_detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

**Option B - Using Anaconda:**
```bash
# Create new conda environment
conda create -n fraud_detection python=3.8

# Activate environment
conda activate fraud_detection
```

### Step 7: Install Dependencies

With your virtual environment activated:
```bash
# Install all required packages
pip install -r requirements.txt

# This will install:
# - pandas, numpy, matplotlib, seaborn
# - scikit-learn, imbalanced-learn
# - jupyter, notebook, ipykernel
```

**Troubleshooting:**
- If you get "pip not found": Install Python from python.org
- If packages fail: Try `pip install --upgrade pip` first
- On Mac, you might need: `pip3 install -r requirements.txt`

### Step 8: Launch Jupyter Notebook

```bash
# Make sure you're in the project folder
cd ~/Desktop/fraud_detection

# Launch Jupyter
jupyter notebook

# This will open your browser automatically
# If not, copy the URL from terminal (usually http://localhost:8888)
```

### Step 9: Run the Notebook

1. **In Jupyter Browser Interface:**
   - Click on `fraud_detection_notebook.ipynb`
   - This opens the notebook

2. **Run Cells Step by Step:**
   - Click on the first cell
   - Press `Shift + Enter` to run cell and move to next
   - Or use menu: Cell → Run All

3. **Expected Output:**
   - Each cell will execute in order
   - You'll see data loading, preprocessing, model training
   - Plots will appear inline
   - Final model will be saved to `models/` folder

### Step 10: Alternative - Run Python Script

If you prefer running as a script instead of notebook:
```bash
# Make sure you're in project folder
cd ~/Desktop/fraud_detection

# Run the script
python fraud_detection.py

# This will:
# - Load and process data
# - Train all models
# - Save plots to figures/ folder
# - Save best model to models/ folder
```

## Verification Checklist

✅ **Before Running:**
- [ ] fraud_detection folder created
- [ ] All 4 files in main folder
- [ ] data/ folder contains creditcard.csv
- [ ] Python environment activated
- [ ] All packages installed successfully

✅ **After Running:**
- [ ] Models saved in models/ folder
- [ ] Plots saved in figures/ folder
- [ ] No error messages in notebook/terminal

## Common Issues & Solutions

**Issue 1: "Module not found" error**
- Solution: Make sure virtual environment is activated and packages installed

**Issue 2: "File not found: creditcard.csv"**
- Solution: Ensure CSV is in data/ folder, not main folder

**Issue 3: Jupyter won't start**
- Solution: Try `pip install --upgrade jupyter notebook`

**Issue 4: Memory error**
- Solution: Close other applications, the dataset is large (144MB)

## Expected Results

After successful execution:
- **Training time:** 5-10 minutes depending on computer
- **Best model:** Random Forest with ~0.87 F1-Score
- **Output files:**
  - `models/best_model.pkl` - Trained model
  - `models/scaler.pkl` - Data scaler
  - `figures/confusion_matrix.png`
  - `figures/roc_curves.png`
  - `figures/model_comparison.png`
  - `figures/feature_importance.png`

## Next Steps

Once everything runs successfully:
1. Review the results in the notebook
2. Try modifying hyperparameters
3. Experiment with different models
4. Add to your GitHub portfolio

## Need Help?

- Python installation: https://www.python.org/downloads/
- Jupyter documentation: https://jupyter.org/documentation
- Kaggle dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

**Pro Tips:**
- Keep virtual environment activated while working
- Save notebook frequently (Ctrl+S)
- Clear output before sharing: Cell → All Output → Clear
- Export as PDF for resume: File → Download as → PDF