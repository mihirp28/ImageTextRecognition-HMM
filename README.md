## Problem
Our goal is to recognize text in an image - e.g., to recognize that **Figure 3** says “It is so ordered.” But the images are noisy, so any particular letter may be difficult to recognize. However, if we make the assumption that these images have English words and sentences, we can use statistical properties of the language to resolve ambiguities. We’ll assume that all the text in our images has the same fixed-width font of the same size. In particular, each letter fits in a box that’s 16 pixels wide and 25 pixels tall. We’ll also assume that our documents only have the 26 uppercase latin characters, the 26 lowercase characters, the 10 digits, spaces, and 7 punctuation symbols, (), .- !?’". Suppose we’re trying to recognize a text string with n characters, so we have n observed variables (the subimage corresponding to each letter) O<sub>1</sub>, . . . , O<sub>n</sub> and n hidden variables, l<sub>1</sub> . . . , l<sub>n</sub>, which are the letters we want to recognize. We’re thus interested in P (l<sub>1</sub>, . . . , l<sub>n</sub> | O<sub>1</sub>, . . . , O<sub>n</sub>). As in part 1 , we can rewrite this using Bayes’ Law, estimate P (O<sub>i</sub> | l<sub>i</sub>) and P (l<sub>i</sub> | l<sub>i−1</sub>) from training data, then use probabilistic inference to estimate the posterior, in order to recognize letters.

![image](https://github.com/user-attachments/assets/a1469347-ac39-43d0-8544-8d1657e7b892)

  python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png

The program should load in the train-image-file, which contains images of letters to use for training (we’ve supplied one for you). It should also load in the text training file, which is simply some text document that is representative of the language (English, in this case) that will be recognized. (The training file from Part 1 could be a good choice). Then, it should use the classifier it has learned to detect the text in test-image-file.png, using (1) the simple Bayes net of Figure 1 and (2) the HMM of Fig 2 with MAP inference (Viterbi). The last two lines of output from your program should be these two results, as follows: 
  python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
  Simple: 1t 1s so orcerec.
  HMM: It is so ordered.


# ImageTextRecognition-HMM

**Goal:** The goal is to use two types of models, Simple Bayes Nets, and Hidden Markov Models (HMM) with map inference, to extract text from a noisy scanned image of a document. To execute the Python code, use the following arguments: python3 image2text.py courier-train.png bc.train testimages/<<image_name>>.png.

**Problem Explanation:** We aim to predict text using two methods: Simple Bayes and HMM. We use initial and transition probabilities for these predictions. To acquire training data, we reuse the bc.train file. For emission probabilities, we compare each training letter to each testing letter, considering the ratio of correct black and white characters relative to the total number of characters. The Simple Bayes net provides optimal results for each hidden state (letter). We then implement initial state probabilities, transition probabilities, and emission probabilities into an HMM. A Viterbi matrix is created and backpropagated to find the optimal output using MAP (Maximum A Posteriori) inference. The final outputs for both Simple Bayes and HMM are printed.
