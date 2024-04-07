import time
import datetime

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, TIMESTAMP

engine = create_engine(
    'mysql+pymysql://root:love29338615@127.0.0.1:3306/PJI', echo=True)
meta = MetaData()
table = Table(
    'message', meta,
    Column('username', String(120), nullable=False),
    Column('content', String(1000)),
    # Column('segment', String(30)),
    # Column('hgb', String(30)),
    # Column('platelet', String(30)),
    # Column('serum_WBC', String(30)),
    # Column('p_t', String(30)),
    # Column('aptt', String(30)),
    # Column('total_cci', String(30)),
    # Column('total_elixhauser_groups_per_record', String(30)),
    # Column('primary_revision_native_hip', String(30)),
    # Column('asa', String(30)),
    # Column('positive_culture', String(30)),
    # Column('Serum_CRP', String(30)),
    # Column('serum_ESR', String(30)),
    # Column('synovial_WBC', String(30)),
    # Column('single_positive_culture', String(30)),
    # Column('synovial_PMN', String(30)),
    # Column('positive_histology', String(30)),
    # Column('pulurence', String(30)),
)
meta.create_all(engine)
conn = engine.connect()

# title = ["no_group", "age", "segment", "hgb", "platelet", "serum_WBC", "p_t", "aptt", "total_cci", "total_elixhauser_groups_per_record",
#          "primary_revision_native_hip", "asa", "positive_culture", "Serum_CRP", "serum_ESR", "synovial_WBC", "single_positive_culture", "synovial_PMN", "positive_histology", "pulurence"]


# pji_csv = open('PJI_Dataset/New_data_x_test.csv', 'r')
# pji_csv_content = pji_csv.read().replace('\r\n', '\n')
# pji_csv_row = pji_csv_content.split('\n')
# for row in pji_csv_row[1:]:
#     columns = row.strip().split(',')
#     obj = dict(zip(title, columns))
#     ins = table.insert(obj)
#     result = conn.execute(ins)
