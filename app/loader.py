import os 
import joblib 

def load_models(cls_path,reg_path):
    if not os.path.exist(cls_path):
        raise FileNotFoundError(f"Classifier not found:{cls_path}")
    if not os.path.exist(reg_path_path):
        raise FileNotFoundError(f"Regression not found:{reg_path}")
    cls = joblib.load(cls_path)
    reg = joblib.load(reg_path)
    return cls, reg
