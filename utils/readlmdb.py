import lmdb
import msgpack
import re
import msgpack_numpy

msgpack_numpy.patch()
from lz4.frame import compress, decompress
from pytorch_pretrained_bert import BertTokenizer
from cytoolz import curry

YON = re.compile(r"^is|^are|^was|^were|^does|^do|^did|^can|^could|^would|^will|^should|^have|^has|^had")
WHAT_IS = re.compile(r"^what (is|was|are|were)")
WHAT_DO = re.compile(r"^what (do|does|did|have|has|had)")
WHAT_STH = re.compile(
    r"^what (color|kind|type|sort|sport|animal|object|country|fruit|material|pattern|hand|number|food|gender|time)")
HOW_MANY = re.compile(r"^how (many|much)")
WHICH = re.compile(r"^which")
WHO = re.compile(r"^who")
WHY = re.compile(r"^why (is|was|are|were)")
WHERE_IS = re.compile(r"^where (is|was|are|were)")
WHERE_DO = re.compile(r"^where (do|does|did|have|has|had)")
HOW_IS = re.compile(r"^how (is|was|are|were)")
HOW_DO = re.compile(r"^how (do|does|did|have|has|had)")
HOW_OLD = re.compile(r"^how old (is|was|are|were)")

"""
将疑问句改为陈述句的函数实现
"""


def convert(question: str) -> str:
    # 首先删掉句尾问号
    input = question
    input = input.rstrip('?')
    input = input.strip()
    # 将第一个字母调为小写
    input = input[0].lower() + input[1:]
    l = input.split()
    output = ""
    # 一般疑问句 -> 陈述句
    # is it xxx ? - > it is [MASK] xxx .
    # is the man xxx ? -> the man is [] xxx
    if re.match(YON, input):
        if l[1] == "the":
            l[0], l[1], l[2] = l[1], l[2], l[0]
            l.insert(3, "[MASK]")
        else:
            l[0], l[1] = l[1], l[0]
            l.insert(2, "[MASK]")
        output = " ".join(l)
    # what is this? -> this is [MASK]
    elif re.match(WHAT_IS, input):
        l.pop(0)
        be = l.pop(0)
        l.append(be)
        l.append("[MASK]")
        output = " ".join(l)
    # what does the sign say? -> the sign say [MASK]
    elif re.match(WHAT_DO, input):
        l.pop(0)
        l.pop(0)
        l.append("[MASK]")
        output = " ".join(l)
    elif re.match(WHAT_STH, input):
        # 首先定位is/was/are/were位置
        index = -1
        for i, x in enumerate(l):
            if x == "is" or x == "was" or x == "are" or x == "were":
                index = i
                break
        if index != -1:
            front = l[:index]  # 截取前部字符串
            l = l[index:]
            front[0] = "[MASK]"
            l.append(l.pop(0))
            l.extend(front)
            output = " ".join(l)
    # how many cats in the shot? -> [MASK] cats in the shot.
    elif re.match(HOW_MANY, input):
        # 一般情况下直接将How many/How much替换为[MASK]
        # 可以在头部加上There are
        # How many wheels does the plane have ?
        l.pop(0)
        l.pop(0)
        l.insert(0, "[MASK]")
        output = " ".join(l)
    elif re.match(WHICH, input):
        l.pop(0)
        l.insert(0, "[MASK]")
        output = " ".join(l)
    elif re.match(WHO, input):
        l.pop(0)
        l.insert(0, "[MASK]")
        output = " ".join(l)
    elif re.match(WHY, input):
        l.pop(0)
        be = l.pop(0)
        l.extend([be, "because", "[MASK]"])
        output = " ".join(l)
    elif re.match(WHERE_IS, input):
        l.pop(0)
        be = l.pop(0)
        l.extend([be, "in", "place", "[MASK]"])
        output = " ".join(l)
    elif re.match(WHERE_DO, input):
        l.pop(0)
        l.pop(0)
        front = "The place where "
        l.extend(["is", "[MASK]"])
        output = front + " ".join(l)
    elif re.match(HOW_IS, input):
        l.pop(0)
        be = l.pop(0)
        l.extend([be, "[MASK]"])
        output = " ".join(l)
    elif re.match(HOW_DO, input):
        l.pop(0)
        l.pop(0)
        l.append("[MASK]")
        output = " ".join(l)
    elif re.match(HOW_OLD, input):
        l.pop(0)
        l.pop(0)
        be = l.pop(0)
        l.extend([be, "[MASK]", "years", "old"])
        output = " ".join(l)
    # --------------------------------------------#
    # 判断output是否为空，以及是否满足上述条件添加了[MASK]
    if len(output) == 0 or output.isspace() or ("[MASK]" not in output):
        output = question.strip() + "The answer is [MASK]"
    return output[0].upper() + output[1:]


tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

# db_folder = "/txt/vqa_devval.db"
# db_folder = "/txt/vqa_vg.db"
# db_folder = "/txt/vqa_test.db"
# db_folder = "/txt/vqa_trainval.db"
db_folder = "/txt/vqa_train.db"

def convertLMDB():
    env = lmdb.open(db_folder, map_size=1099511627776)
    txn = env.begin(write=True)
    for key, value in txn.cursor():
        dic = msgpack.loads(decompress(value), raw=False)  # 值可读化

        question = dic['question']
        statement = convert(question)

        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(statement))

        dic['input_ids'] = input_ids
        revalue = compress(msgpack.dumps(dic, use_bin_type=True))  # 序列化

        txn.put(key, revalue)

    txn.commit()
    env.close()


def test():
    with open("../vocab.txt", "r") as f:
        vocab = f.read().splitlines()
    env = lmdb.open(db_folder, map_size=1099511627776)
    txn = env.begin(write=False)
    for key, value in txn.cursor():
        dic = msgpack.loads(decompress(value), raw=False)  # 值可读化
        input_ids = dic['input_ids']
        question = dic['question']
        statement = " ".join([vocab[id] for id in input_ids])
        print(question + " || " + statement)
    env.close()

convertLMDB()
test()