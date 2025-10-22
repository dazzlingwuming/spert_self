'''
spert模型预测器
'''
from sympy.printing.pytorch import torch
from transformers import BertConfig, BertTokenizer
from spert import util
from spert import prediction, models
from spert.input_reader import JsonInputReader, JsonPredictionInputReader
from spert.models import load_model
from spert.prediction import text_predictions
from spert.sampling import create_eval_sample, collate_fn_padding
from spert.util import creat_text2token


class SpertPredictor(object):
    def __init__(self ,types_path,model_type,model_path,cache_path,tokenizer_path,max_span_size = 25 ,
                 max_pairs = 100,prop_drop=0.1,size_embedding = 15,rel_filter_threshold = 0.7):
        self.types_path = types_path
        self.model_type = model_type
        self.model_path = model_path
        self.cache_path = cache_path
        self.tokenizer_path = tokenizer_path
        self.do_lower_case = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_span_size = max_span_size
        self.max_pairs = max_pairs
        self.prop_drop=prop_drop
        self.size_embedding =size_embedding
        self.freeze_transformer = False
        self.rel_filter_threshold = rel_filter_threshold
        self.spacy_model = None
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path,
                                                        do_lower_case=self.do_lower_case,
                                                        cache_dir=self.cache_path)
        self.input_reader = JsonPredictionInputReader(self.types_path, self.tokenizer,
                                        max_span_size=self.max_span_size,
                                        spacy_model=self.spacy_model)
        self.model =load_model(self , self.tokenizer , self.input_reader)

        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text: str):
        #获取解析后的文本
        doc  = self.input_reader.parse_token(text)

        #得到实体对象
        batch = create_eval_sample(doc, max_span_size = self.max_span_size)
        batch = util.to_device(collate_fn_padding([batch]) , self.device)

        #模型预测
        self.model.eval()
        # run model (forward pass)
        result = self.model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                       entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                       entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                       inference=True)
        entity_clf, rel_clf, rels = result

        # convert predictions 结果解析
        predictions = prediction.convert_predictions(entity_clf,
                                                     rel_clf,
                                                     rels,
                                                     batch,
                                                     self.rel_filter_threshold,
                                                     self.input_reader
                                                     )

        batch_pred_entities, batch_pred_relations = predictions
        #这里只预测一个样本，所以取第0个
        predictions = text_predictions([doc] , batch_pred_entities , batch_pred_relations )
        return predictions[0]
        print("xxx")


if __name__ == "__main__":
    _nlp = creat_text2token()
    print(_nlp("Hello world! This is a test."))