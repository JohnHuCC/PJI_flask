import csv

title = ["no_group", "no", "group", "name", "ctno", "csn", "PJI_revision_date",
         "Date_of_first_surgery", "Date_of_last_surgery", "primary_revision_native_hip",
         "acute_chronic", "asa", "laterality", "joint", "gender", "age",
         "height", "weight", "bmi", "serum_ESR", "serum_WBC", "segment",
         "hgb", "platelet", "p_t", "aptt", "fibrinogen", "d_dimer",
         "Serum_CRP", "cr", "ast", "alt", "positive_histology",
         "synovial_WBC", "synovial_Neutrophil", "turbidity", "color",
         "lymphocyte", "monocyte", "eosinophil",
         "synovial_leukocyte_esterase", "total_core", "icm_2",
         "minor_icm_criteria_total", "icm_1", "minor_msis_criteria_total",
         "msis_final_classification", "positive_culture", "sinus_tract",
         "pulurence", "single_positive_culture", "total_cci",
         "congestive_heart_failure", "cardiac_arrhythmia",
         "valvular_disease", "pulmonary_circulation_disorders",
         "peripheral_vascular_disorders", "hypertension_uncomplicated",
         "hypertension_complicated", "paralysis",
         "other_neurological_disorders", "chronic_pulmonary_disease",
         "diabetes_uncomplicated", "diabetes_complicated",
         "hypothyroidism", "renal_failure", "liver_disease",
         "peptic_ulcer_disease_excluding_bleeding", "aids_hiv",
         "lymphoma", "metastatic_cancer",
         "solid_tumor_without_metastasis",
         "rheumatoid_arthritis_collagen", "coagulopathy",
         "obesity", "weight_loss", "fluid_electrolyte_disorders",
         "blood_loss_anemia", "deficiency_anemia", "alcohol_abuse",
         "drug_abuse", "psychoses", "depression",
         "total_elixhauser_groups_per_record"]

filename = "Revision_PJI_test_2.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=title)
    writer.writeheader()

    # 產生兩筆測試資料
    row1 = {col: "" for col in title}
    row1["no_group"] = "G001"
    row1["no"] = "1"
    row1["group"] = "A"
    row1["name"] = "Test Patient 1"

    row2 = {col: "" for col in title}
    row2["no_group"] = "G002"
    row2["no"] = "2"
    row2["group"] = "B"
    row2["name"] = "Test Patient 2"

    writer.writerow(row1)
    writer.writerow(row2)

print(f"{filename} generated successfully!")