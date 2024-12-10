import pandas as pd
df = pd.read_excel("strongsort_fastsam_out.xlsx")
df.to_latex("strongsort_fastsam_out.tex")

