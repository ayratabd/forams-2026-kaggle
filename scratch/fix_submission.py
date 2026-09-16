import pandas as pd

def fix_submission(input_path, output_path):
    df = pd.read_csv(input_path, keep_default_na=False)
    
    LABEL_MAP = {
        "Globigerina_bulloides": "cl00",
        "Globigerinella_calida": "cl01",
        "Globigerinella_siphonifera": "cl02",
        "Globigerinita_glutinata": "cl03",
        "Globigerinoides_conglobatus": "cl04",
        "Globigerinoides_ruber": "cl05",
        "Globigerinoides_rubescens": "cl06",
        "Globorotalia_crassaformis": "cl07",
        "Globorotalia_inflata": "cl08",
        "Globorotalia_menardii": "cl09",
        "Globorotalia_scitula": "cl10",
        "Globorotalia_truncatulinoides": "cl11",
        "Neogloboquadrina_dutertrei": "cl12",
        "Neogloboquadrina_incompta": "cl13",
        "Neogloboquadrina_pachyderma": "cl14",
        "Orbulina_universa": "cl15",
        "Pulleniatina_obliquiloculata": "cl16",
        "Turborotalita_quinqueloba": "cl17"
    }
    
    fixed_rows = []
    for _, row in df.iterrows():
        fname = row["filename"]
        cp = row["centerpoint"]
        
        if not cp or pd.isna(cp) or cp.strip() == "":
            # Add a dummy prediction to bypass "null values" Kaggle error
            fixed_rows.append({"filename": fname, "centerpoint": "cl00;0.00;0.00;0.00"})
            continue
            
        tokens = cp.split(";")
        new_tokens = []
        for i in range(0, len(tokens), 4):
            cls_name = tokens[i]
            x, y, z = tokens[i+1], tokens[i+2], tokens[i+3]
            if cls_name in LABEL_MAP:
                new_tokens.extend([LABEL_MAP[cls_name], z, y, x])
            else:
                new_tokens.extend([cls_name, z, y, x])
                
        fixed_rows.append({"filename": fname, "centerpoint": ";".join(new_tokens)})
        
    pd.DataFrame(fixed_rows).to_csv(output_path, index=False)
    print(f"Fixed submission saved to {output_path}")

if __name__ == "__main__":
    fix_submission("submissions/submission_v1.csv", "submission.csv")
