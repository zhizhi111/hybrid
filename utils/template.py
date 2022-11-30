import re

"""
1手工模板
1.1统计各个类型的样本数
1.2提示句子,补充完整，完形填空式
1.3内容上加一些先验的知识：定义、约束、解释性说明、描述

2软模板
2.1冻结原模型，至训练软标签
2.2在模型上直接微调
"""


class WTemplate:
    def __init__(self):
        # ^限定开头
        self.YON = re.compile(r"^Is|^Are|^Was|^Were|^Does|^Do|^Did|^Can|^Could|^Would|^Will|^Should|^Have|^Has|^Had") #
        self.COLOR = re.compile(r"^What color") #
        self.KIND = re.compile(r"^What (kind|sort)") #
        self.TYPE = re.compile(r"^What type") #
        self.TIME = re.compile(r"^What time") #
        self.SPORT = re.compile(r"^What sport") #
        self.BRAND = re.compile(r"^What brand") #
        self.HM = re.compile(r"^How many") #
        self.HMU = re.compile(r"^How much") #
        self.OBJ = re.compile(r"^Which") ##
        self.WHO = re.compile(r"^Who") #
        self.WHY = re.compile(r"^Why") #
        self.WHERE = re.compile(r"^Where") #
        self.NAME = re.compile(r"^What is the name") #
        self.ANIMAL = re.compile(r"^What animal") #
        self.OBJECT = re.compile(r"^What object") #
        self.COUNTRY = re.compile(r"^What country") #
        self.HOW = re.compile(r"^How (is|was|were)")#
        self.FRUIT = re.compile(r"^What fruit") #
        self.MATERIAL = re.compile(r"^What material") #
        self.PATTERN = re.compile(r"^What pattern") #
        self.HAND = re.compile(r"^What hand") #
        self.NUMBER = re.compile(r"^What number") #
        self.AGE = re.compile(r"^How old") #
        self.FOOD = re.compile(r"^What food") #
        self.GENDER = re.compile(r"^What gender") #
        # self.hybrid_prompt = [f"[SEP] Answer: [MASK] . Please answer  yes or no.Attribute: ",
        #                       f"[SEP] Answer: [MASK] . Please answer  color such as white, blue, red.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  a time point such as 8:00 or time duration such as afternoon, night.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  a subtype of a type such as cat from animal, pizza from food.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  a sport such as tennis, baseball.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  a name of company such as nike, apple, dell.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  an integer such as 1, 2, 3.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  something such as lot, 0, little.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  something about the asked object such as left, right, north.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the person asked such as man, woman, boy.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  a reason such as safety, yes, raining.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  a location such as beach, outside, table.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the name of the asked object such as big ben, pizza, united.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the type of the animal such as cat, dog, horse.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the name of the object asked such as frisbee, umbrella, kite.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the name of the country asked such as usa, england, china.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  an adjective word such as sunny, clear, good.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the name of the fruit asked such as banana, orange, anpple.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  what material the object asked is made of such as wood, metal, brick.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the pattern of the object asked such as strips, solid, plaid.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the hand asked such as left, right, both.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the age of the person asked such as 10, 1, old.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the name of the food asked such as pizza, sandwich, cake.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the gender of person asked such as male, famale, woman.Attribute:",
        #                       f"[SEP] Answer: [MASK] . Please answer  the question asked.Attribute:"]

    def pre_question(self, q):
        if re.match(self.YON, q):
            return f"[SEP] Answer: [MASK] . Please answer  yes or no."
        elif re.match(self.COLOR, q):
            return f"[SEP] Answer: [MASK] . Please answer  color such as white, blue, red."
        elif re.match(self.TIME, q):
            return f"[SEP] Answer: [MASK] . Please answer  a time point such as 8:00 or time duration such as afternoon, night."
        elif re.match(self.KIND, q) or re.match(self.TYPE, q):
            return f"[SEP] Answer: [MASK] . Please answer  a subtype of a type such as cat from animal, pizza from food."
        elif re.match(self.SPORT, q):
            return f"[SEP] Answer: [MASK] . Please answer  a sport such as tennis, baseball."
        elif re.match(self.BRAND, q):
            return f"[SEP] Answer: [MASK] . Please answer  a name of company such as nike, apple, dell."
        elif re.match(self.HM, q) or re.match(self.NUMBER, q):
            return f"[SEP] Answer: [MASK] . Please answer  an integer such as 1, 2, 3."
        elif re.match(self.HMU, q):
            return f"[SEP] Answer: [MASK] . Please answer  something such as lot, 0, little."
        elif re.match(self.OBJ, q):
            return f"[SEP] Answer: [MASK] . Please answer  something about the asked object such as left, right, north."
        elif re.match(self.WHO, q):
            return f"[SEP] Answer: [MASK] . Please answer  a person such as man, woman, boy."
        elif re.match(self.WHY, q):
            return f"[SEP] Answer: [MASK] . Please answer  a reason such as safety, yes, raining."
        elif re.match(self.WHERE, q):
            return f"[SEP] Answer: [MASK] . Please answer  a location such as beach, outside, table."
        elif re.match(self.NAME, q):
            return f"[SEP] Answer: [MASK] . Please answer  a name such as big ben, pizza, united."
        elif re.match(self.ANIMAL, q):
            return f"[SEP] Answer: [MASK] . Please answer  an animal such as cat, dog, horse."
        elif re.match(self.OBJECT, q):
            return f"[SEP] Answer: [MASK] . Please answer  an object such as frisbee, umbrella, kite."
        elif re.match(self.COUNTRY, q):
            return f"[SEP] Answer: [MASK] . Please answer  a country asked such as usa, england, china."
        elif re.match(self.HOW, q):
            return f"[SEP] Answer: [MASK] . Please answer  an adjective word such as sunny, clear, good."
        elif re.match(self.FRUIT, q):
            return f"[SEP] Answer: [MASK] . Please answer  fruit such as banana, orange, anpple."
        elif re.match(self.MATERIAL, q):
            return f"[SEP] Answer: [MASK] . Please answer  material such as wood, metal, brick."
        elif re.match(self.PATTERN, q):
            return f"[SEP] Answer: [MASK] . Please answer  a pattern such as strips, solid, plaid."
        elif re.match(self.HAND, q):
            return f"[SEP] Answer: [MASK] . Please answer  a hand such as left, riht, both."
        elif re.match(self.AGE, q):
            return f"[SEP] Answer: [MASK] . Please answer  age such as 10, 1, old."
        elif re.match(self.FOOD, q):
            return f"[SEP] Answer: [MASK] . Please answer  food such as pizza, sandwich, cake."
        elif re.match(self.GENDER, q):
            return f"[SEP] Answer: [MASK] . Please answer  a gender such as male, famale, woman."
        else:
            return f"[SEP] Answer: [MASK] . Please answer  the question."

    def pre_question1(self, q):
        if re.match(self.YON, q):
            return f"[SEP] Answer: [MASK] . Please answer  yes or no."
        elif re.match(self.COLOR, q):
            return f"[SEP] Answer: [MASK] . Please answer  color such as white, blue, red."
        elif re.match(self.TIME, q):
            return f"[SEP] Answer: [MASK] . Please answer  time such as 8:00, afternoon, night."
        elif re.match(self.KIND, q) or re.match(self.TYPE, q):
            return f"[SEP] Answer: [MASK] . Please answer  a subtype such as cat from animal, pizza from food."
        elif re.match(self.SPORT, q):
            return f"[SEP] Answer: [MASK] . Please answer  a sport such as tennis, baseball."
        elif re.match(self.BRAND, q):
            return f"[SEP] Answer: [MASK] . Please answer  a name of company such as nike, apple, dell."
        elif re.match(self.HM, q) or re.match(self.NUMBER, q):
            return f"[SEP] Answer: [MASK] . Please answer  an integer such as 1, 2, 3."
        elif re.match(self.HMU, q):
            return f"[SEP] Answer: [MASK] . Please answer  something such as lot, 0, little."
        elif re.match(self.OBJ, q):
            return f"[SEP] Answer: [MASK] . Please answer  something about the asked object such as left, right, north."
        elif re.match(self.WHO, q):
            return f"[SEP] Answer: [MASK] . Please answer  the person asked such as man, woman, boy."
        elif re.match(self.WHY, q):
            return f"[SEP] Answer: [MASK] . Please answer  a reason such as safety, yes, raining."
        elif re.match(self.WHERE, q):
            return f"[SEP] Answer: [MASK] . Please answer  a location such as beach, outside, table."
        elif re.match(self.NAME, q):
            return f"[SEP] Answer: [MASK] . Please answer  the name of the asked object such as big ben, pizza, united."
        elif re.match(self.ANIMAL, q):
            return f"[SEP] Answer: [MASK] . Please answer  the type of the animal such as cat, dog, horse."
        elif re.match(self.OBJECT, q):
            return f"[SEP] Answer: [MASK] . Please answer  the name of the object asked such as frisbee, umbrella, kite."
        elif re.match(self.COUNTRY, q):
            return f"[SEP] Answer: [MASK] . Please answer  the name of the country asked such as usa, england, china."
        elif re.match(self.HOW, q):
            return f"[SEP] Answer: [MASK] . Please answer  an adjective word such as sunny, clear, good."
        elif re.match(self.FRUIT, q):
            return f"[SEP] Answer: [MASK] . Please answer  the name of the fruit asked such as banana, orange, anpple."
        elif re.match(self.MATERIAL, q):
            return f"[SEP] Answer: [MASK] . Please answer  what material the object asked is made of such as wood, metal, brick."
        elif re.match(self.PATTERN, q):
            return f"[SEP] Answer: [MASK] . Please answer  the pattern of the object asked such as strips, solid, plaid."
        elif re.match(self.HAND, q):
            return f"[SEP] Answer: [MASK] . Please answer  the hand asked such as left, riht, both."
        elif re.match(self.AGE, q):
            return f"[SEP] Answer: [MASK] . Please answer  the age of the person asked such as 10, 1, old."
        elif re.match(self.FOOD, q):
            return f"[SEP] Answer: [MASK] . Please answer  the name of the food asked such as pizza, sandwich, cake."
        elif re.match(self.GENDER, q):
            return f"[SEP] Answer: [MASK] . Please answer  the gender of person asked such as male, famale, woman."
        else:
            return f"[SEP] Answer: [MASK] . Please answer  the question asked."

    def mixed_prompt1(self, q):
        if re.match(self.YON, q):
            return (0, f"[SEP] Answer: [MASK] .")
        elif re.match(self.COLOR, q):
            return (1, f"[SEP] Answer: [MASK] .")
        elif re.match(self.TIME, q):
            return (2, f"[SEP] Answer: [MASK] .")
        elif re.match(self.KIND, q) or re.match(self.TYPE, q):
            return (3, f"[SEP] Answer: [MASK] .")
        elif re.match(self.SPORT, q):
            return (4, f"[SEP] Answer: [MASK] .")
        elif re.match(self.BRAND, q):
            return (5, f"[SEP] Answer: [MASK] .")
        elif re.match(self.HM, q) or re.match(self.NUMBER, q):
            return (6, f"[SEP] Answer: [MASK] .")
        elif re.match(self.HMU, q):
            return (7, f"[SEP] Answer: [MASK] .")
        elif re.match(self.OBJ, q):
            return (8, f"[SEP] Answer: [MASK] .")
        elif re.match(self.WHO, q):
            return (9, f"[SEP] Answer: [MASK] .")
        elif re.match(self.WHY, q):
            return (10, f"[SEP] Answer: [MASK] .")
        elif re.match(self.WHERE, q):
            return (11, f"[SEP] Answer: [MASK] .")
        elif re.match(self.NAME, q):
            return (12, f"[SEP] Answer: [MASK] .")
        elif re.match(self.ANIMAL, q):
            return (13, f"[SEP] Answer: [MASK] .")
        elif re.match(self.OBJECT, q):
            return (14, f"[SEP] Answer: [MASK] .")
        elif re.match(self.COUNTRY, q):
            return (15, f"[SEP] Answer: [MASK] .")
        elif re.match(self.HOW, q):
            return (16, f"[SEP] Answer: [MASK] .")
        elif re.match(self.FRUIT, q):
            return (17, f"[SEP] Answer: [MASK] .")
        elif re.match(self.MATERIAL, q):
            return (18, f"[SEP] Answer: [MASK] .")
        elif re.match(self.PATTERN, q):
            return (19, f"[SEP] Answer: [MASK] .")
        elif re.match(self.HAND, q):
            return (20, f"[SEP] Answer: [MASK] .")
        elif re.match(self.AGE, q):
            return (21, f"[SEP] Answer: [MASK] .")
        elif re.match(self.FOOD, q):
            return (22, f"[SEP] Answer: [MASK] .")
        elif re.match(self.GENDER, q):
            return (23, f"[SEP] Answer: [MASK] .")
        else:
            return (24, f"[SEP] Answer: [MASK] .")

    def mixed_prompt2(self, q):
        if re.match(self.YON, q):
            return (0, f"[SEP] Answer: [MASK] . Please answer  yes or no")
        elif re.match(self.COLOR, q):
            return (1, f"[SEP] Answer: [MASK] . Please answer  color such as ")
        elif re.match(self.TIME, q):
            return (2, f"[SEP] Answer: [MASK] . Please answer  a time point such as  ")
        elif re.match(self.KIND, q) or re.match(self.TYPE, q):
            return (3, f"[SEP] Answer: [MASK] . Please answer  a subtype of a type such as ")
        elif re.match(self.SPORT, q):
            return (4, f"[SEP] Answer: [MASK] . Please answer  a sport such as ")
        elif re.match(self.BRAND, q):
            return (5, f"[SEP] Answer: [MASK] . Please answer  a name of company such as ")
        elif re.match(self.HM, q) or re.match(self.NUMBER, q):
            return (6, f"[SEP] Answer: [MASK] . Please answer  an integer such as ")
        elif re.match(self.HMU, q):
            return (7, f"[SEP] Answer: [MASK] . Please answer  something such as ")
        elif re.match(self.OBJ, q):
            return (8, f"[SEP] Answer: [MASK] . Please answer  something about the asked object such as ")
        elif re.match(self.WHO, q):
            return (9, f"[SEP] Answer: [MASK] . Please answer  the person asked such as ")
        elif re.match(self.WHY, q):
            return (10, f"[SEP] Answer: [MASK] . Please answer  a reason such as ")
        elif re.match(self.WHERE, q):
            return (11, f"[SEP] Answer: [MASK] . Please answer  a location such as ")
        elif re.match(self.NAME, q):
            return (12, f"[SEP] Answer: [MASK] . Please answer  the name of the asked object such as ")
        elif re.match(self.ANIMAL, q):
            return (13, f"[SEP] Answer: [MASK] . Please answer  the type of the animal such as ")
        elif re.match(self.OBJECT, q):
            return (14, f"[SEP] Answer: [MASK] . Please answer  the name of the object asked such as ")
        elif re.match(self.COUNTRY, q):
            return (15, f"[SEP] Answer: [MASK] . Please answer  the name of the country asked such as ")
        elif re.match(self.HOW, q):
            return (16, f"[SEP] Answer: [MASK] . Please answer  an adjective word such as ")
        elif re.match(self.FRUIT, q):
            return (17, f"[SEP] Answer: [MASK] . Please answer  the name of the fruit asked such as ")
        elif re.match(self.MATERIAL, q):
            return (18, f"[SEP] Answer: [MASK] . Please answer  what material the object asked is made of such as ")
        elif re.match(self.PATTERN, q):
            return (19, f"[SEP] Answer: [MASK] . Please answer  the pattern of the object asked such as ")
        elif re.match(self.HAND, q):
            return (20, f"[SEP] Answer: [MASK] . Please answer  the hand asked such as ")
        elif re.match(self.AGE, q):
            return (21, f"[SEP] Answer: [MASK] . Please answer  the age of the person asked such as ")
        elif re.match(self.FOOD, q):
            return (22, f"[SEP] Answer: [MASK] . Please answer  the name of the food asked such as ")
        elif re.match(self.GENDER, q):
            return (23, f"[SEP] Answer: [MASK] . Please answer  the gender of person asked such as ")
        else:
            return (24, f"[SEP] Answer: [MASK] . Please answer  the question asked.")

    def weak_prompt(self, q):
        if re.match(self.YON, q):
            return 0
        elif re.match(self.COLOR, q):
            return 1
        elif re.match(self.TIME, q):
            return 2
        elif re.match(self.KIND, q) or re.match(self.TYPE, q):
            return 3
        elif re.match(self.SPORT, q):
            return 4
        elif re.match(self.BRAND, q):
            return 5
        elif re.match(self.HM, q) or re.match(self.NUMBER, q):
            return 6
        elif re.match(self.HMU, q):
            return 7
        elif re.match(self.OBJ, q):
            return 8
        elif re.match(self.WHO, q):
            return 9
        elif re.match(self.WHY, q):
            return 10
        elif re.match(self.WHERE, q):
            return 11
        elif re.match(self.NAME, q):
            return 12
        elif re.match(self.ANIMAL, q):
            return 13
        elif re.match(self.OBJECT, q):
            return 14
        elif re.match(self.COUNTRY, q):
            return 15
        elif re.match(self.HOW, q):
            return 16
        elif re.match(self.FRUIT, q):
            return 17
        elif re.match(self.MATERIAL, q):
            return 18
        elif re.match(self.PATTERN, q):
            return 19
        elif re.match(self.HAND, q):
            return 20
        elif re.match(self.AGE, q):
            return 21
        elif re.match(self.FOOD, q):
            return 22
        elif re.match(self.GENDER, q):
            return 23
        else:
            return 24

    def mixed_prompt(self, q):
        if re.match(self.YON, q):
            return (0, f"[SEP] Answer: [MASK] . Please answer  yes or no")
        elif re.match(self.COLOR, q):
            return (1, f"[SEP] Answer: [MASK] . Please answer  color such as white, blue, red")
        elif re.match(self.TIME, q):
            return (2, f"[SEP] Answer: [MASK] . Please answer  a time point such as 8:00 or time duration such as afternoon, night")
        elif re.match(self.KIND, q) or re.match(self.TYPE, q):
            return (3, f"[SEP] Answer: [MASK] . Please answer  a subtype of a type such as cat from animal, pizza from food")
        elif re.match(self.SPORT, q):
            return (4, f"[SEP] Answer: [MASK] . Please answer  a sport such as tennis, baseball")
        elif re.match(self.BRAND, q):
            return (5, f"[SEP] Answer: [MASK] . Please answer  a name of company such as nike, apple, dell")
        elif re.match(self.HM, q) or re.match(self.NUMBER, q):
            return (6, f"[SEP] Answer: [MASK] . Please answer  an integer such as 1, 2, 3")
        elif re.match(self.HMU, q):
            return (7, f"[SEP] Answer: [MASK] . Please answer  something such as lot, 0, little")
        elif re.match(self.OBJ, q):
            return (8, f"[SEP] Answer: [MASK] . Please answer  something about the asked object such as left, right, north")
        elif re.match(self.WHO, q):
            return (9, f"[SEP] Answer: [MASK] . Please answer  the person asked such as man, woman, boy")
        elif re.match(self.WHY, q):
            return (10, f"[SEP] Answer: [MASK] . Please answer  a reason such as safety, yes, raining")
        elif re.match(self.WHERE, q):
            return (11, f"[SEP] Answer: [MASK] . Please answer  a location such as beach, outside, table")
        elif re.match(self.NAME, q):
            return (12, f"[SEP] Answer: [MASK] . Please answer  the name of the asked object such as big ben, pizza, united")
        elif re.match(self.ANIMAL, q):
            return (13, f"[SEP] Answer: [MASK] . Please answer  the type of the animal such as cat, dog, horse")
        elif re.match(self.OBJECT, q):
            return (14, f"[SEP] Answer: [MASK] . Please answer  the name of the object asked such as frisbee, umbrella, kite")
        elif re.match(self.COUNTRY, q):
            return (15, f"[SEP] Answer: [MASK] . Please answer  the name of the country asked such as usa, england, china")
        elif re.match(self.HOW, q):
            return (16, f"[SEP] Answer: [MASK] . Please answer  an adjective word such as sunny, clear, good")
        elif re.match(self.FRUIT, q):
            return (17, f"[SEP] Answer: [MASK] . Please answer  the name of the fruit asked such as banana, orange, anpple")
        elif re.match(self.MATERIAL, q):
            return (18, f"[SEP] Answer: [MASK] . Please answer  what material the object asked is made of such as wood, metal, brick")
        elif re.match(self.PATTERN, q):
            return (19, f"[SEP] Answer: [MASK] . Please answer  the pattern of the object asked such as strips, solid, plaid")
        elif re.match(self.HAND, q):
            return (20, f"[SEP] Answer: [MASK] . Please answer  the hand asked such as left, riht, both")
        elif re.match(self.AGE, q):
            return (21, f"[SEP] Answer: [MASK] . Please answer  the age of the person asked such as 10, 1, old")
        elif re.match(self.FOOD, q):
            return (22, f"[SEP] Answer: [MASK] . Please answer  the name of the food asked such as pizza, sandwich, cake")
        elif re.match(self.GENDER, q):
            return (23, f"[SEP] Answer: [MASK] . Please answer  the gender of person asked such as male, famale, woman")
        else:
            return (24, f"[SEP] Answer: [MASK] . Please answer  the question asked.")
