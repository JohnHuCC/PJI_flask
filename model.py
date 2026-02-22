from db import db
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    session_hash = db.Column(db.String(150), nullable=True)

    def insert(username, password):
        db.session.add(User(name=username, password=password))
        db.session.commit()

    def get_by_name(name):
        """
        get_by_name : String -> User
        """
        return User.query.filter_by(name=name).first()

    def get_by_uid(uid):
        return User.query.filter_by(session_hash=uid).first()

    def check_password(self, passwd):
        """
        密碼驗證，驗證使用者輸入的密碼跟資料庫內的加密密碼是否相符
        :param password: 使用者輸入的密碼
        :return: Bool
        """
        return self.password == passwd

    def load_user(user_id):
        return User.query.get(int(user_id))


class RevisionPJI(db.Model):
    no_group = db.Column(db.String(120), unique=True,
                         nullable=False, primary_key=True)
    no = db.Column(db.String(120), unique=True, nullable=False)
    group = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(120), unique=True, nullable=False)
    computed_tomography_number = db.Column(
        db.String(30), unique=True, nullable=False)
    ctno = db.Column(db.String(30), unique=True)
    csn = db.Column(db.String(30), unique=True)
    PJI_revision_date = db.Column(db.String(30))
    Date_of_first_surgery = db.Column(db.String(30))
    Date_of_last_surgery = db.Column(db.String(30))
    primary_revision_native_hip = db.Column(db.String(30))
    acute_chronic = db.Column(db.String(30))
    asa = db.Column(db.String(30))
    laterality = db.Column(db.String(30))
    joint = db.Column(db.String(30))
    gender = db.Column(db.String(30))
    age = db.Column(db.String(30))
    # m
    height = db.Column(db.String(30))
    # kg
    weight = db.Column(db.String(30))
    bmi = db.Column(db.String(30))
    serum_ESR = db.Column(db.String(30))
    serum_WBC = db.Column(db.String(30))
    # ( % )
    segment = db.Column(db.String(30))
    hgb = db.Column(db.String(30))
    platelet = db.Column(db.String(30))
    p_t = db.Column(db.String(30))
    aptt = db.Column(db.String(30))
    fibrinogen = db.Column(db.String(30))
    d_dimer = db.Column(db.String(30))
    Serum_CRP = db.Column(db.String(30))
    # (B)
    cr = db.Column(db.String(30))
    ast = db.Column(db.String(30))
    alt = db.Column(db.String(30))
    positive_histology = db.Column(db.String(30))
    synovial_WBC = db.Column(db.String(30))
    synovial_Neutrophil = db.Column(db.String(30))
    turbidity = db.Column(db.String(30))
    color = db.Column(db.String(30))
    lymphocyte = db.Column(db.String(30))
    monocyte = db.Column(db.String(30))
    eosinophil = db.Column(db.String(30))
    synovial_leukocyte_esterase = db.Column(db.String(30))
    total_score = db.Column(db.String(30))
    icm_2 = db.Column(db.String(30))
    minor_icm_criteria_total = db.Column(db.String(30))
    icm_1 = db.Column(db.String(30))
    minor_msis_criteria_total = db.Column(db.String(30))
    msis_final_classification = db.Column(db.String(30))
    positive_culture = db.Column(db.String(30))
    sinus_tract = db.Column(db.String(30))
    pulurence = db.Column(db.String(30))
    single_positive_culture = db.Column(db.String(30))
    total_cci = db.Column(db.String(30))
    congestive_heart_failure = db.Column(db.String(30))
    cardiac_arrhythmia = db.Column(db.String(30))
    valvular_disease = db.Column(db.String(30))
    pulmonary_circulation_disorders = db.Column(db.String(30))
    peripheral_vascular_disorders = db.Column(db.String(30))
    hypertension_uncomplicated = db.Column(db.String(30))
    hypertension_complicated = db.Column(db.String(30))
    paralysis = db.Column(db.String(30))
    other_neurological_disorders = db.Column(db.String(30))
    chronic_pulmonary_disease = db.Column(db.String(30))
    diabetes_uncomplicated = db.Column(db.String(30))
    diabetes_complicated = db.Column(db.String(30))
    hypothyroidism = db.Column(db.String(30))
    renal_failure = db.Column(db.String(30))
    liver_disease = db.Column(db.String(30))
    peptic_ulcer_disease_excluding_bleeding = db.Column(db.String(30))
    aids_hiv = db.Column(db.String(30))
    lymphoma = db.Column(db.String(30))
    metastatic_cancer = db.Column(db.String(30))
    solid_tumor_without_metastasis = db.Column(db.String(30))
    rheumatoid_arthritis_collagen = db.Column(db.String(30))
    coagulopathy = db.Column(db.String(30))
    obesity = db.Column(db.String(30))
    weight_loss = db.Column(db.String(30))
    fluid_electrolyte_disorders = db.Column(db.String(30))
    blood_loss_anemia = db.Column(db.String(30))
    deficiency_anemia = db.Column(db.String(30))
    alcohol_abuse = db.Column(db.String(30))
    drug_abuse = db.Column(db.String(30))
    psychoses = db.Column(db.String(30))
    depression = db.Column(db.String(30))
    total_elixhauser_groups_per_record = db.Column(db.String(30))
