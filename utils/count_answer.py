import pyarrow as pa
import pyarrow.parquet as pq

data_path = ["/data/wjx/vqa/data/train_val.parquet", "/data/wjx/vqa/data/train2014.parquet"]

tb = pa.concat_tables([pq.read_table(t) for t in data_path]).to_pandas()
templates = [
    "What color","What kind","What sort","What type","What time",
    "What sport","What brand","How many","How much","Which",
    "Who","Why","Where","What is the name","What animal",
    "What object","What country","How is","How was","How were",
    "What fruit","What material","What pattern","What hand",
    "What number","How old","What food","What gender"
]

for template in templates:
    x = tb.question.str.contains(template)
    y = [i for i in range(len(x)) if x[i]]
    z = tb.iloc[y]
    print(f"---------start of {template}------------------------------")
    print(z.answer.value_counts()[:10])
    print(f"---------end of {template}------------------------------")
