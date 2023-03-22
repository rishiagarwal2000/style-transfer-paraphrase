from scipy.stats import kendalltau
import tqdm
import collections
import itertools
import numpy as np
import torch
import argparse
import subprocess
import re
import sys
#sys.path.append('fairseq')
#sys.path.append('style_paraphrase')
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel
# from style_paraphrase.evaluation.similarity.test_sim import find_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from style_paraphrase.evaluation.similarity.sim_models import WordAveraging
from style_paraphrase.evaluation.similarity.sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm

def _detokenize(x):
	x = x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")
	return x
def _style_classifier_label_fn(model,label):
		#model can be RobertaModels data memebers here
		return model.task.label_dictionary.string(
			[label + model.task.target_dictionary.nspecial]
		)

class Evaluator():
	def __init__(self,gpu=True,device='cuda:0'):
		self.gpu=gpu
		self.style_classifier=RobertaModel.from_pretrained(
			'style_paraphrase/style_classify/saved_models/save_0',
			checkpoint_file='checkpoint_best.pt',
			data_name_or_path='datasets/dialouge_dataset-bin'
		)

		if gpu:
			self.style_classifier.to(device)
		self.style_classifier.eval()
		self.op_style_label=0 if _style_classifier_label_fn(self.style_classifier,0)=='dialogue' else 1

		self.batch_size=1
		self.device=device
		
		self.style_loss_model= AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
		self.style_loss_tokenizer=AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
		self.style_loss_tokenizer.pad_token = self.style_loss_tokenizer.eos_token
		self.style_loss_model.eval()
		if gpu:
			self.style_loss_model.to(device)

		self.acceptability_classifier=RobertaModel.from_pretrained(
			'style_paraphrase/evaluation/fluency/cola_classifier',
			checkpoint_file='checkpoint_best.pt',
			data_name_or_path='style_paraphrase/evaluation/fluency/cola_classifier/cola-bin'
		)
		if gpu:
			self.acceptability_classifier.to(device)
		self.acceptability_classifier.eval()
		self.acceptable_label=0 if _style_classifier_label_fn(self.acceptability_classifier,0)=='acceptable' else 1
		self.unk_bpe = self.acceptability_classifier.bpe.encode(" <unk>").strip()

		self.sim_tok = TreebankWordTokenizer()

		model = torch.load('style_paraphrase/evaluation/similarity/sim/sim.pt',map_location=torch.device(device))
		state_dict = model['state_dict']
		vocab_words = model['vocab_words']
		args = model['args']
		# turn off gpu
		self.model = WordAveraging(args, vocab_words)
		self.model.load_state_dict(state_dict, strict=True)
		self.sp = spm.SentencePieceProcessor()
		self.sp.Load('style_paraphrase/evaluation/similarity/sim/sim.sp.30k.model')
		self.model.eval()

	def make_example(self,sentence, model):
	    sentence = sentence.lower()
	    sentence = " ".join(self.sim_tok.tokenize(sentence))
	    sentence = self.sp.EncodeAsPieces(sentence)
	    wp1 = Example(" ".join(sentence))
	    wp1.populate_embeddings(model.vocab)
	    return wp1

	def find_similarity(self,s1, s2):
	    with torch.no_grad():
	        s1 = [self.make_example(x, self.model) for x in s1]
	        s2 = [self.make_example(x, self.model) for x in s2]
	        wx1, wl1, wm1 = self.model.torchify_batch(s1)
	        wx2, wl2, wm2 = self.model.torchify_batch(s2)
	        scores = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
	        return [x.item() for x in scores]
	
	def style_score(self,out_sents):
		#returns a tensor of scores
		style_scores=[]
		for i in range(0, len(out_sents), self.batch_size):
			sds=out_sents[i:i + self.batch_size]
			sds = [self.style_classifier.bpe.encode(_detokenize(sd)) for sd in sds]
			batch = collate_tokens(
			[self.style_classifier.task.source_dictionary.encode_line("<s> " + sd + " </s>", append_eos=False) for sd in sds], pad_idx=1
			)

			batch = batch[:, :512]
			with torch.no_grad():
				predictions = self.style_classifier.predict('sentence_classification_head', batch.long())
			
			probabilities=torch.exp(predictions)# they sum to 1 with this much only cuzz model gives log probs
			style_scores.append(probabilities[:,self.op_style_label])
		return torch.cat(style_scores,0).to(torch.device('cpu'))

	def style_score_loss(self,out_sents):
		rewards=[]
		for sent in out_sents:
			encoding = self.style_loss_tokenizer.encode(sent, return_tensors="pt",truncation=True).to(self.device)
			with torch.no_grad():
				loss=self.style_loss_model.forward(encoding,labels=encoding).loss
			rewards.append(loss)
		return torch.tensor(rewards).squeeze()


	def acceptability_score(self,out_sents):
		#returns a tensor of scores
		acc_scores=[]
		for i in range(0, len(out_sents), self.batch_size):
			sds=out_sents[i:i + self.batch_size]
			sds = [self.acceptability_classifier.bpe.encode(_detokenize(sd)) for sd in sds]
			batch = collate_tokens(
			[self.acceptability_classifier.task.source_dictionary.encode_line("<s> " + sd + " </s>", append_eos=False) for sd in sds], pad_idx=1
			)

			batch = batch[:, :512]
			with torch.no_grad():
				predictions = self.acceptability_classifier.predict('sentence_classification_head', batch.long())
			
			probabilities=torch.exp(predictions)# they sum to 1 with this much only cuzz model gives log probs
			acc_scores.append(probabilities[:,self.acceptable_label])

		return torch.cat(acc_scores,0).to(torch.device('cpu'))

	def similarity_score(self,in_sents,out_sents):
		#input to the generator and correspondng output
		#returns a tensor of scores
		assert len(in_sents)==len(out_sents)
		sim_scores=[]
		for i in range(0, len(in_sents), self.batch_size):
			sim_scores.extend(
				self.find_similarity(in_sents[i:i + self.batch_size], out_sents[i:i + self.batch_size])
			)
		
		return torch.tensor(sim_scores)

	def unicode_score(self,out_sents):
		score=[0.0]*len(out_sents)
		for i,sent in enumerate(out_sents):
			for char in sent:
				if ord(char)>=65533:
					score[i]=-10
					break
		return torch.tensor(score)

	def len_score(self,in_sents,out_sents):
		scores=[]
		def len_rep(x):
			return len(x.replace(' ',''))
		for a,b in zip(in_sents,out_sents):
			scores.append((len_rep(b)/len_rep(a)))
		return torch.tensor(scores)

	def combined_score(self,in_sents,out_sents):
		return (self.style_score(out_sents)+self.similarity_score(in_sents,out_sents)+self.acceptability_score(out_sents))/3

	def mult_combined(self,in_sents,out_sents):
		return (self.style_score(out_sents)*self.similarity_score(in_sents,out_sents)*self.acceptability_score(out_sents))
		# return -self.style_score_loss(out_sents)/6+2*self.similarity_score(in_sents,out_sents)+self.len_score(in_sents,out_sents)



if __name__=='__main__':
	ev=Evaluator(gpu=True)
	for tf in ['basic_wquad_len_t5_inject_milestone399.txt','basic_wquad_len_t5_inject_milestone599.txt','norm_dialogue_t5_inject_milestone199.txt','norm_dialogue_t5_inject_milestone399.txt',
	'test_gpt2_milestone_op.txt','test_outputs_GPT2_milestone_dialogpt_loss199.txt','test_outputs_GPT2_milestone_dialogpt_loss399.txt','test_news_only.txt.paraphrase']:

		print('***',tf,'***')
		
		with open('/home/ubuntu/style-transfer-paraphrase/datasets/dialouge_dataset/test_news_only.txt','r') as fr:
			in_sents=fr.readlines()

		with open(f'/home/ubuntu/style-transfer-paraphrase/datasets/dialouge_dataset/{tf}','r') as fr:
			out_sents=fr.readlines()

		style_score=torch.mean(ev.style_score(out_sents))
		acc_score=torch.mean(ev.acceptability_score(out_sents))
		fleuncy_score=torch.mean(ev.similarity_score(in_sents,out_sents))
		len_score=torch.mean(ev.len_score(in_sents,out_sents))
		unicode_score=torch.mean(ev.unicode_score(out_sents))
		style_score_loss=torch.mean(1/ev.style_score_loss(out_sents))
		comb_score=(style_score+acc_score+fleuncy_score)/3
		J_score=torch.mean(ev.mult_combined(in_sents,out_sents))
		print(f'style score:{style_score}')
		print(f'acc score:{acc_score}')
		print(f'fluency score:{fleuncy_score}')
		print(f'combined score:{comb_score}')
		print(f'J score:{J_score}')
		print(f'len_score:{len_score}')
		print(f'unicode_score:{unicode_score}')
		print(f'DialoGPT style_score_loss:{style_score_loss}')

	 #tr=['� � � � � � � � � � � � � � � � � � � � � � � � � � � � � � � � � ']
	# tr=['ϧ']
	#print(ev.style_score(tr))
	#print(ev.acceptability_score(tr))
	 #print(ev.fleuncy_score(tr))

	# ev.combined_score
	# print(ev.similarity_score(tr))
	# path='outputs/baselines/unmt_shakespeare/transfer_entire_test.txt'

	# from datasets import load_dataset
	# # dataset_news = load_dataset("cnn_dailymail",'3.0.0')
	# # sents=datasetnews['test']['highlights']
	# dataset_dialogue2 = load_dataset("empathetic_dialogues")
	# sents2=dataset_dialogue2['test']['utterance']
	# print()
	
	# with open(path,'r') as fr:
	# 	sents=fr.readlines()
	
	# print(ev.combined_score(sents[:10]))
	# print(ev.combined_score(sents2[:10],sents2[:10]))
	


