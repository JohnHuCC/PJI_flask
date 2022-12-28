import time
import datetime

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, TIMESTAMP

engine = create_engine(
    'mysql+pymysql://root:love29338615@127.0.0.1:3306/PJI', echo=True)
meta = MetaData()
table = Table(
    'revision_pji_part', meta,
    Column('no_group', String(120), nullable=False, primary_key=True),
    Column('no', String(120),  nullable=False),
    Column('group', String(10), nullable=False),
    Column('name', String(120), nullable=False),
    Column('ctno', String(30)),
    Column('csn', String(30)),
    Column('PJI_revision_date', String(30)),
    Column('Date_of_first_surgery', String(30)),
    Column('Date_of_last_surgery', String(30)),
    Column('primary_revision_native_hip', String(30)),
    Column('acute_chronic', String(30)),
    Column('asa', String(30)),
    Column('laterality', String(30)),
    Column('joint', String(30)),
    Column('gender', String(30)),
    Column('age', String(30)),
    Column('height', String(30)),
    Column('weight', String(30)),
    Column('bmi', String(30)),
    Column('serum_ESR', String(30)),
    Column('serum_WBC', String(30)),
    Column('segment', String(30)),
    Column('hgb', String(30)),
    Column('platelet', String(30)),
    Column('p_t', String(30)),
    Column('aptt', String(30)),
    Column('fibrinogen', String(30)),
    Column('d_dimer', String(30)),
    Column('Serum_CRP', String(30)),
    Column('cr', String(30)),
    Column('ast', String(30)),
    Column('alt', String(30)),
    Column('positive_histology', String(30)),
    Column('synovial_WBC', String(30)),
    Column('synovial_Neutrophil', String(30)),
    Column('turbidity', String(30)),
    Column('color', String(30)),
    Column('lymphocyte', String(30)),
    Column('monocyte', String(30)),
    Column('eosinophil', String(30)),
    Column('synovial_leukocyte_esterase', String(30)),
    Column('total_core', String(30)),
    Column('icm_2', String(30)),
    Column('minor_icm_criteria_total', String(30)),
    Column('icm_1', String(30)),
    Column('minor_msis_criteria_total', String(30)),
    Column('msis_final_classification', String(30)),
    Column('positive_culture', String(30)),
    Column('sinus_tract', String(30)),
    Column('pulurence', String(30)),
    Column('single_positive_culture', String(30)),
    Column('total_cci', String(30)),
    Column('congestive_heart_failure', String(30)),
    Column('cardiac_arrhythmia', String(30)),
    Column('valvular_disease', String(30)),
    Column('pulmonary_circulation_disorders', String(30)),
    Column('peripheral_vascular_disorders', String(30)),
    Column('hypertension_uncomplicated', String(30)),
    Column('hypertension_complicated', String(30)),
    Column('paralysis', String(30)),
    Column('other_neurological_disorders', String(30)),
    Column('chronic_pulmonary_disease', String(30)),
    Column('diabetes_uncomplicated', String(30)),
    Column('diabetes_complicated', String(30)),
    Column('hypothyroidism', String(30)),
    Column('renal_failure', String(30)),
    Column('liver_disease', String(30)),
    Column('peptic_ulcer_disease_excluding_bleeding', String(30)),
    Column('aids_hiv', String(30)),
    Column('lymphoma', String(30)),
    Column('metastatic_cancer', String(30)),
    Column('solid_tumor_without_metastasis', String(30)),
    Column('rheumatoid_arthritis_collagen', String(30)),
    Column('coagulopathy', String(30)),
    Column('obesity', String(30)),
    Column('weight_loss', String(30)),
    Column('fluid_electrolyte_disorders', String(30)),
    Column('blood_loss_anemia', String(30)),
    Column('deficiency_anemia', String(30)),
    Column('alcohol_abuse', String(30)),
    Column('drug_abuse', String(30)),
    Column('psychoses', String(30)),
    Column('depression', String(30)),
    Column('total_elixhauser_groups_per_record', String(30)),
)
meta.create_all(engine)
conn = engine.connect()

title = ["no_group", "no", "group", "name", "ctno", "csn", "PJI_revision_date", "Date_of_first_surgery", "Date_of_last_surgery", "primary_revision_native_hip", "acute_chronic", "asa", "laterality", "joint", "gender", "age", "height", "weight", "bmi", "serum_ESR", "serum_WBC", "segment", "hgb", "platelet", "p_t", "aptt", "fibrinogen", "d_dimer", "Serum_CRP", "cr", "ast", "alt", "positive_histology", "synovial_WBC", "synovial_Neutrophil", "turbidity", "color", "lymphocyte", "monocyte", "eosinophil", "synovial_leukocyte_esterase", "total_core", "icm_2", "minor_icm_criteria_total", "icm_1", "minor_msis_criteria_total", "msis_final_classification", "positive_culture", "sinus_tract", "pulurence", "single_positive_culture",
         "total_cci", "congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease", "pulmonary_circulation_disorders", "peripheral_vascular_disorders", "hypertension_uncomplicated", "hypertension_complicated", "paralysis", "other_neurological_disorders", "chronic_pulmonary_disease", "diabetes_uncomplicated", "diabetes_complicated", "hypothyroidism", "renal_failure", "liver_disease", "peptic_ulcer_disease_excluding_bleeding", "aids_hiv", "lymphoma", "metastatic_cancer", "solid_tumor_without_metastasis", "rheumatoid_arthritis_collagen", "coagulopathy", "obesity", "weight_loss", "fluid_electrolyte_disorders", "blood_loss_anemia", "deficiency_anemia", "alcohol_abuse", "drug_abuse", "psychoses", "depression", "total_elixhauser_groups_per_record"]


pji_csv = open('/Users/johnnyhu/Desktop/Revision_PJI_part.csv', 'r')
pji_csv_content = pji_csv.read().replace('\r\n', '\n')
pji_csv_row = pji_csv_content.split('\n')
for row in pji_csv_row[1:]:
    columns = row.strip().split(',')
    obj = dict(zip(title, columns))
    ins = table.insert(obj)
    result = conn.execute(ins)
