from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import re

model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')
model.eval()

def clean(text):
    text = re.sub(r'http\S+', " ", text)
    text = re.sub(r'@\w+',' ',text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'<.*?>',' ', text)
    return text

def classification (title, body):
  user_input = tokenizer(title,
                        body,
                        padding=True,
                        truncation=True,
                        return_tensors="pt")

  with torch.no_grad():
      outputs = model(user_input['input_ids'], token_type_ids=None, 
                      attention_mask=user_input['attention_mask'])
  logits = outputs[0][0][0]
  result = 1 if float(logits) > 0.5 else 0
  return result


# example
title = "Donald Trump Sends Out Embarrassing New Year s Eve Message; This is Disturbing"
body = "Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  The former reality show star had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted.    will be a great year for America! As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year.   will be a great year for America!  Donald J. Trump ( ) December  ,  Trump s tweet went down about as welll as you d expect.What kind of president sends a New Year s greeting like this despicable, petty, infantile gibberish? Only Trump! His lack of decency won t even allow him to rise above the gutter long enough to wish the American citizens a happy new year!  Bishop Talbert Swan ( ) December  ,  no one likes you  Calvin ( ) December  ,  Your impeachment would make   a great year for America, but I ll also accept regaining control of Congress.  Miranda Yaver ( ) December  ,  Do you hear yourself talk? When you have to include that many people that hate you you have to wonder? Why do the they all hate me?  Alan Sandoval ( ) December  ,  Who uses the word Haters in a New Years wish??  Marlene ( ) December  ,  You can t just say happy new year?  Koren pollitt ( ) December  ,  Here s Trump s New Year s Eve tweet from  .Happy New Year to all, including to my many enemies and those who have fought me and lost so badly they just don t know what to do. Love!  Donald J. Trump ( ) December  ,  This is nothing new for Trump. He s been doing this for years.Trump has directed messages to his  enemies  and  haters  for New Year s, Easter, Thanksgiving, and the anniversary of  / . pic.twitter.com/ FPAe KypA  Daniel Dale ( ) December  ,  Trump s holiday tweets are clearly not presidential.How long did he work at Hallmark before becoming President?  Steven Goodine ( ) December  ,  He s always been like this . . . the only difference is that in the last few years, his filter has been breaking down.  Roy Schulze ( ) December  ,  Who, apart from a teenager uses the term haters?  Wendy ( ) December  ,  he s a fucking   year old  Who Knows ( ) December  ,  So, to all the people who voted for this a hole thinking he would change once he got into power, you were wrong!  -year-old men don t change and now he s a year older.Photo by Andrew Burton/Getty Images."
a = classification(title, body)
print(a)