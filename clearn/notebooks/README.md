# Notebooks

If you want to make an ipython notebook to play with the main scripts, 
* run it in the `/clearn` directory (There's some awkwardness with the way we're handling imports and file paths. You could do iPython dark magic and run the notebook from anywhere, but your life will be easier if you just run it from the clearn package root.)
* Add this magic block to bring in our modules (I do not doubt that there is a nicer way to handle this. But I don't know what it is.):
```
import imp
clearn = imp.load_source('clearn', '__init__.py')
munge = imp.load_source('munge', 'munge.py')
predict = imp.load_source('predict', 'predict.py')
evaluate = imp.load_source('evaluate', 'evaluate.py')
```
* If you want to push your notebook up for posterity, save it in this folder for safekeeping.
