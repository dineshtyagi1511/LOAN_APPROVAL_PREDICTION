# Loan Approval Project - Two Stage ML Application

## OverView : 
Two stage model : 
1. Classifiy applicant as Approved / Rejected
2. IF Approved, predict the loan amount.

## Quickstack (local)
1. Create venv:
- uv venv
- uv pip install -r requirements.txt
2. Put your trained `stage_1...pkl` and `stage_2...pkl` files in `models\`.
3. Run locally : 
- Interface/UI : `uv run streamlit run streamlit_app.py`
- CLI : `uv run python main.py`

## Config
See `config.yaml` for runtime paramters (models paths)

## To install/freeze additional libraries using UV :
```bash
uv pip install -r requirements.txt
uv pip freeze > requirements.txt
```

## Git instructions :
```bash
git init
git add .
git commit -m "mesasage"
git remote add origin https://url_of_you_git_repo.git
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## NOTE : 
- Make sure the version used to create the model is same as your local environment where you are testing the main.py or streamlit app.
- We are using python 3.12 for our virtual environment.


