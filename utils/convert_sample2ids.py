from pytorch_pretrained_bert import BertTokenizer
import secrets

class sample2ids():
    def __init__(self):
        self.topics = [
            " judge ",
            " color ",
            " time ",
            " type ",
            " sport ",
            " company ",
            " integer ",
            " quantity ",
            " direction ",
            " person ",
            " reason ",
            " location ",
            " name ",
            " animal ",
            " object ",  # frisbee
            " country ",
            " modification ",  #
            " fruit ",
            " material ",
            " pattern ",  # plaid
            " hand ",
            " age ",
            " food ",
            " gender ",
            " answer "
        ]
        self.samples = [
            " yes no unknown ",
            " white blue red black  ",
            " afternoon night evening noon  ",
            " cat pizza wood bus  ",
            " tennis baseball skiing soccer  ",
            " Nike Apple Dell Wilson  ",
            " 1 2 3 0 ",
            " little lot half all ",
            " left right north ",
            " man woman boy girl ",
            " safety yes rainy ",
            " beach outside table airport ",
            " ben pizza united unknown ",
            " cat dog horse ",
            " umbrella kite ",  # frisbee
            " USA England China Canada ",
            " Sunny clear good rainy ",  #
            " banana orange apple  ",
            " wood metal brick ",
            " strips solid floral ",  # plaid
            " right left both ",
            " 1 old 3 ",
            " pizza sandwich cake ",
            " male female girl ",
            " unknown unknown unknown "
        ]
        # self.samples = [
        #     " yes no unknown ",
        #     " white blue red black green brown yellow gray orange pink ",
        #     " afternoon night evening noon morning daytime winter summer ",
        #     " cat pizza wood bus sheep oak bathroom ",
        #     " tennis baseball skiing soccer skate surfing ",
        #     " Nike Apple Dell Wilson Samsung Sony",
        #     " 1 2 3 0 4 5 6 7 8",
        #     " little lot half all ",
        #     " left right north ",
        #     " man woman boy girl ",
        #     " safety yes rainy ",
        #     " beach outside table airport zoo right street nowhere ",
        #     " ben pizza united unknown ",
        #     " cat dog horse elephant sheep cow bear ",
        #     " umbrella kite ",  # frisbee
        #     " USA England China Canada India Japan France ",
        #     " Sunny clear good rainy ",  #
        #     " banana orange apple  ",
        #     " wood metal brick ",
        #     " strips solid floral ",  # plaid
        #     " right left both ",
        #     " 1 old 3 ",
        #     " pizza sandwich cake ",
        #     " male female girl ",
        #     " unknown unknown unknown "
        # ]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
        self.topics_ids = []
        self.sample_ids = []
        for topic in self.topics:
            topic_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(topic))
            self.topics_ids.append(topic_id[0])
        for sample in self.samples:
            sample_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sample))
            self.sample_ids.append(sample_id)

    """
    根据输入索引（分类结果）返回对应Topic的id
    """

    def get_topic(self, index):
        return self.topics_ids[index]

    """
    根据输入索引（分类结果）返回对应的Sample id
    从存在的Sample id中随机选取3条返回 —— 构造更多的正样例
    """

    def get_sample(self, index):
        sampls = self.sample_ids[index]  # List
        assert len(sampls) >= 3
        # 随机选择3个正样例返回
        return secrets.SystemRandom().sample(sampls, 3)
