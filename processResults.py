import pandas as pd


def main():
    header = ["Predictor", "Species", "Read length", "Error (%)", "Sensitivity", "Specificity", "Accuracy"]
    df = pd.read_csv("predResults.csv", names=header, sep='\t')

    species = set(df["Species"].tolist())
    lengths = [100,200,400,700]
    err = [0,1,3]
    for spec in species:
        print("| Species | Read length | Error (%) | Sn (RFD)| Sn (FGS) | Sp (RFD)| Sp (FGS) | Acc (RFD)| Acc (FGS) |")
        print("|---|---|---|---|---|---|---|---|---|")

        #print(spec)
        includeSpecies = True
        for l in lengths:
            #print(l)
            includeLen = True
            for e in err:
                if l == 700 and e == 3:
                    e = 0.5
                
                idx_rfd = df.index[(df["Predictor"] == "RFD") & (df["Species"] == spec) & (df["Read length"] == l) & (df["Error (%)"] == e)].tolist()[0]

                idx_fgs = df.index[(df["Predictor"] == "FGS") & (df["Species"] == spec) & (df["Read length"] == l) & (df["Error (%)"] == e)].tolist()[0]



                if includeSpecies:
                    row = f"| {spec}"
                    includeSpecies = False
                else:
                    row = "| "
                
                if includeLen:
                    row += f"| {l}"
                    includeLen = False
                else:
                    row += "| "
                
                row += f"| {e} "
                
                sn_rfd = df.iloc[idx_rfd]["Sensitivity"]
                sp_rfd = df.iloc[idx_rfd]["Specificity"]
                ac_rfd = df.iloc[idx_rfd]["Accuracy"]
                sn_fgs = df.iloc[idx_fgs]["Sensitivity"]
                sp_fgs = df.iloc[idx_fgs]["Specificity"]
                ac_fgs = df.iloc[idx_fgs]["Accuracy"]

                if sn_fgs > sn_rfd:
                    sn_fgs = f"**{sn_fgs:.{3}}**"
                    sn_rfd = f"{sn_rfd:.{3}}"
                else:
                    sn_rfd = f"**{sn_rfd:.{3}}**"
                    sn_fgs = f"{sn_fgs:.{3}}"

                if sp_fgs > sp_rfd:
                    sp_fgs = f"**{sp_fgs:.{3}}**"
                    sp_rfd = f"{sp_rfd:.{3}}"
                else:
                    sp_rfd = f"**{sp_rfd:.{3}}**"
                    sp_fgs = f"{sp_fgs:.{3}}"

                if ac_fgs > ac_rfd:
                    ac_fgs = f"**{ac_fgs:.{3}}**"
                    ac_rfd = f"{ac_rfd:.{3}}"
                else:
                    ac_rfd = f"**{ac_rfd:.{3}}**"
                    ac_fgs = f"{ac_fgs:.{3}}"

                row += f" | {sn_rfd} | {sn_fgs} | {sp_rfd} | {sp_fgs} | {ac_rfd} | {ac_fgs} |"

                print(row)
        print("\n")


if __name__ == '__main__':
    main()