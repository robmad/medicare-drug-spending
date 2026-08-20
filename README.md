   # Medicare Drug Spending Prediction
   
   Building an end-to-end ML pipeline to predict Medicare Part D drug spending.
   
   ## Project Structure
   - `data/`: Raw and processed datasets (local only, see .gitignore)
   - `notebooks/`: EDA and analysis
   - `src/`: Training, inference, monitoring code
   - `terraform/`: AWS infrastructure (Phase 2)
   
   ## Getting Started
```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
```
   
   ## Status
   - [x] Phase 1.1: Project structure
   - [ ] Phase 1.2: Data exploration
   - [ ] Phase 1.3: Feature engineering