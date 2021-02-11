# GenerativeSongTitlesAndMusic
Generating song titles using the GPT-2, and generating music through deep learning neural networks.

## Group Members
Samantha Conde and Tamara Duplantis
UCSC CMPM 202, Winter 2021, Project 2

This project contains two examples of generative code:
(1) generating song titles through GPT2 (adapted from Max Woolf's gpt-2-simple colab notebook https://github.com/minimaxir/gpt-2-simple)
(2) generating music through deep learning neural networks (adapted from ...)


# (1) Generating Song Titles with GPT2 
Navigate to the ''TitleGeneration'' Folder

## Example Output Code
Here's an example of five titles generated from GPT-2 (and with a cleaned up text file of the musical, Hamilton), generated with the prefix of "Song of ".

*Song of Thorns Throwing And Angelica Angelica Angel*

*Song of Rose who is always by your side*

*Song of Thorns From your sister Angelica*

*Song of Thorns When Thats the time Now everyone give it up for the maid*

*Song of Thoth Thats it Thats it Thats it*


## Instructions on how to run code
Create a text file of the desired text you wish to influence the title generation.

Pull github repo and ensure you have python 3 installed.

Run on terminal (perferably with ananconda with a python 3 environment):

Navigate to the repo folder and convert the Jupyter Notebook files to google colab. In google collab, initalize your environment with:
```sh
%tensorflow_version 1.x
!pip install -q tensorflow==1.14
!pip install -q gpt-2-simple
import gpt_2_simple as gpt2
from datetime import datetime
from google.colab import files
```

Then to train GPT-2, download one of the three models:
```sh
gpt2.download_gpt2(model_name="124M")
```
Create a reference to your training corpus:
```sh
file_name = "fileWithSampleTextToInfluenceTitleGeneration.txt"
```
And finally, tune GPT-2 with the following:
```sh
sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=1000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=200,
              save_every=500
              )
```
Save the trained model using:
```sh
gpt2.copy_checkpoint_to_gdrive(run_name='run1')
```

Now to generate titles, use the following code. Load the trained model checkpoint:
```sh
gpt2.copy_checkpoint_from_gdrive(run_name='run1')
```
Initalize the session:
```sh
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='run1')
```
And generate song title through:
```sh
gpt2.generate(sess,
              length=15,
              temperature=0.7,
              prefix="Song of ",
              nsamples=5,
              batch_size=5
              )
```


## Rules/Constraints
Parameters presented by Woolf:
```sh
Other optional-but-helpful parameters for gpt2.finetune:

restore_from: Set to fresh to start training from the base GPT-2, or set to latest to restart training from an existing checkpoint.
sample_every: Number of steps to print example output
print_every: Number of steps to print training progress.
learning_rate: Learning rate for the training. (default 1e-4, can lower to 1e-5 if you have <1MB input data)
run_name: subfolder within checkpoint to save the model. This is useful if you want to work with multiple models (will also need to specify run_name when loading the model)
overwrite: Set to True if you want to continue finetuning an existing model (w/ restore_from='latest') without creating duplicate copies.

Other optional-but-helpful parameters for gpt2.generate and friends:

length: Number of tokens to generate (default 1023, the maximum)
temperature: The higher the temperature, the crazier the text (default 0.7, recommended to keep between 0.7 and 1.0)
top_k: Limits the generated guesses to the top k guesses (default 0 which disables the behavior; if the generated output is super crazy, you may want to set top_k=40)
top_p: Nucleus sampling: limits the generated guesses to a cumulative probability. (gets good results on a dataset with top_p=0.9)
truncate: Truncates the input text until a given sequence, excluding that sequence (e.g. if truncate='<|endoftext|>', the returned text will include everything before the first <|endoftext|>). It may be useful to combine this with a smaller length if the input texts are short.
include_prefix: If using truncate and include_prefix=False, the specified prefix will not be included in the returned text.
```


## Process for creating output
See intructions for how to run code
- gpt2.generate generates a batch of nsamples

To generate good output, tweak the temperature parameter to find a balance between your training corpus and the pretrained model of GPT-2. The higher the temperature, the more likely the generated text will be to your corpus (which may add a little grammatical craziness).






# (2) Generating Music with Deep Learning Neural Networks
Navigate to the ''MusicGeneration'' Folder

## Example Output Code
TBD

## Instructions on how to run code
TBD

## Rules/Constraints
TBD

## Process for creating output
TBD

