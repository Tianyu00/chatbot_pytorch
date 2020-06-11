# chatbot_pytorch

# Chatbot 1

## Introduction
Use PyTorch to build a deep learning Sequence to Sequence chatbot with Luong attention. 

The bot is trained on [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).

Reference:
- [Torch](https://pytorch.org)
- [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## Train and talk
You can download the 'chatbot1' folder and go to that folder. Use
```
python chatbot.py
```
to train the model and talk to it in the command line. (If you do not have GPU it would be super slow. It is not recommended to do so.)

You can also go to kaggle and use its GPU to train this bot. See this [notebook](https://www.kaggle.com/tianyuz/chatbot-pytorch?scriptVersionId=35999214).

## Examples
Example results:

Training:
```
Iteration: 1000; Percent complete: 5.0%; Average loss: 4.5071
Iteration: 2000; Percent complete: 10.0%; Average loss: 4.1025
Iteration: 3000; Percent complete: 15.0%; Average loss: 3.9217
Iteration: 4000; Percent complete: 20.0%; Average loss: 3.7354
Iteration: 5000; Percent complete: 25.0%; Average loss: 3.5486
Iteration: 6000; Percent complete: 30.0%; Average loss: 3.3527
Iteration: 7000; Percent complete: 35.0%; Average loss: 3.2122
Iteration: 8000; Percent complete: 40.0%; Average loss: 3.0620
Iteration: 9000; Percent complete: 45.0%; Average loss: 2.8899
Iteration: 10000; Percent complete: 50.0%; Average loss: 2.6947
......
Iteration: 40000; Percent complete: 80.0%; Average loss: 0.7385
Iteration: 41000; Percent complete: 82.0%; Average loss: 0.7483
Iteration: 42000; Percent complete: 84.0%; Average loss: 0.7271
Iteration: 43000; Percent complete: 86.0%; Average loss: 0.7125
Iteration: 44000; Percent complete: 88.0%; Average loss: 0.7123
Iteration: 45000; Percent complete: 90.0%; Average loss: 0.6883
Iteration: 46000; Percent complete: 92.0%; Average loss: 0.6908
Iteration: 47000; Percent complete: 94.0%; Average loss: 0.6874
Iteration: 48000; Percent complete: 96.0%; Average loss: 0.6865
Iteration: 49000; Percent complete: 98.0%; Average loss: 0.6960
Iteration: 50000; Percent complete: 100.0%; Average loss: 0.6880
```

Conversation:
```
> hello
Bot:  how ? juliet
> how are you
Bot:  fine i m fine .
> good
Bot:  good night .
> good night
Bot:  good night .
> all right
Bot:  go like you .
> I don't know what you are talking about
Bot:  i know what i to
> yes??
Bot:  do you like ? ?
> no
Bot:  well let me asking you all the bathroom .
```

Not the most clever bot (You can try using different settings.). Also notice the bot is trained on data [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), so the response is always a little dramatic.
