# MLB Project 4 Main - Fine-Tuning a Language Model for Sentiment Analysis

---

## 1) What this project is about

You'll learn how to **fine-tune a pre-trained language model** for sentiment analysis on product reviews. This project teaches you how to adapt powerful AI models to perform specific tasks by training them on domain-specific data.

You'll use a provided notebook (`MLB_Project_4_Main_FineTuning_Template.ipynb`) that walks you through:
- Loading and preprocessing datasets for fine-tuning
- Tokenizing text data for transformer models
- Configuring training arguments and hyperparameters
- Fine-tuning DistilBERT on Amazon product reviews
- Evaluating model performance with multiple metrics
- Making predictions on new reviews
- Saving and loading your fine-tuned model

By the end, you'll understand how transfer learning works in NLP, how to adapt pre-trained models to new tasks, and how to evaluate classification models using industry-standard metrics.

---

## 2) What is Fine-Tuning (in plain English)

**Fine-tuning** is like teaching a smart student a new specialty. The model already knows language (from pre-training), but you're teaching it your specific task.

Here's how it works:
1. **Start with a pre-trained model**: Use a model that already understands language (like DistilBERT)
2. **Prepare your dataset**: Gather examples of your specific task (product reviews + sentiment labels)
3. **Adapt the model**: Train the model on your data, adjusting its weights slightly
4. **Evaluate performance**: Test how well it performs on unseen examples
5. **Deploy**: Use your fine-tuned model to make predictions

### Why it's useful
- **Much faster** than training from scratch (minutes vs. weeks)
- **Requires less data** than training a model from zero (thousands vs. millions of examples)
- **Better performance** than generic models on your specific task
- **Cost-effective** - leverages existing knowledge

### Key concepts you'll learn
- **Transfer learning**: Using knowledge from one task to improve performance on another
- **Tokenization**: Converting text into numerical tokens models can process
- **Training loop**: How models learn from data through backpropagation
- **Evaluation metrics**: Accuracy, precision, recall, and F1 score
- **Hyperparameters**: Learning rate, batch size, epochs, and how they affect training

---

## 3) The components (what you'll build)

This project has several key components:

### Data Preprocessing
- Load the Amazon product reviews dataset
- Combine title and content fields
- Split data into train/validation/test sets (70/15/15)
- Handle text cleaning and preparation

### Tokenization Pipeline
- Use DistilBERT's tokenizer to convert text to tokens
- Apply padding and truncation for consistent input length
- Create batched datasets for efficient training

### Model Configuration
- Load pre-trained DistilBERT for sequence classification
- Configure for binary sentiment classification (negative/positive)
- Set up data collator for dynamic padding

### Training System
- Define custom metrics (accuracy, precision, recall, F1)
- Configure training arguments (learning rate, batch size, epochs)
- Use Hugging Face Trainer for the training loop
- Implement evaluation after each epoch

### Prediction Interface
- Create a prediction function for new reviews
- Calculate confidence scores using softmax
- Build an interactive classifier for testing

### Models and Data Used
- **Base Model**: `distilbert-base-uncased` (66M parameters, fast and efficient)
- **Dataset**: Amazon Polarity (product reviews with binary sentiment)
- **Training samples**: 5,000 (subset for faster training)

---

## 4) Project files

- **`MLB_Project_FineTuning_Template.ipynb`** (the notebook you'll complete)
- **`MLB_Project_FineTuning_AnswerKey.ipynb`** (complete solution for reference)
- **This README (`README_FineTuning_Sentiment.md`)**: instructions + context

---

## 5) How to open and run in **Google Colab** (recommended)

### Upload the notebook file directly
1. Download the template notebook from Canvas
2. Go to <https://colab.research.google.com/>
3. Click **Upload** → select `MLB_Project_FineTuning_Template.ipynb`
4. **IMPORTANT**: Enable GPU acceleration
   - Click **Runtime → Change runtime type**
   - Set **Hardware accelerator** to **T4 GPU** (or any available GPU)
   - Click **Save**
5. The first cell will install required packages automatically:
   ```python
   !pip install transformers datasets torch scikit-learn accelerate evaluate -q
   ```
6. Click **Runtime → Run all** and follow the TODOs in the notebook

### Why GPU is important
- **CPU training**: ~30-45 minutes per epoch (2+ hours total)
- **GPU training**: ~2-3 minutes per epoch (10-15 minutes total)
- Colab provides free GPU access!

### Running locally (optional)
If you want to run on your own computer with GPU:
1. Install Python 3.8 or higher
2. Install PyTorch with CUDA: Visit <https://pytorch.org/> for your system
3. Install dependencies: `pip install transformers datasets scikit-learn accelerate evaluate`
4. Run: `jupyter notebook MLB_Project_FineTuning_Template.ipynb`

---

## 6) Important: Training time and hardware requirements

**⚠️ GPU is strongly recommended for this project!**

### Expected training times:

**With GPU (Colab T4 - FREE):**
- Total training time: ~10-15 minutes
- Per epoch: ~2-3 minutes
- First run (downloading model): +5 minutes

**With CPU (NOT recommended):**
- Total training time: ~2-3 hours
- Per epoch: ~30-45 minutes
- May cause timeout in Colab

### First run will be slow
The first time you run the notebook, it needs to download:
- The DistilBERT model (~250 MB)
- The Amazon dataset (~80 MB)
- This takes 5-10 minutes depending on internet speed
- After that, everything runs from cache

### Memory requirements
- **GPU**: 4GB VRAM minimum (Colab T4 has 16GB - plenty!)
- **CPU RAM**: 8GB minimum, 16GB recommended
- The dataset and model together use ~2-3GB

---

## 7) How we'll grade (submission criteria)

To receive credit:

1. **Complete the notebook**  
   - Fill in all the `# TODO:` sections with working code
   - The notebook should **run end-to-end** without errors
   - It should produce:
     - Successfully loaded and preprocessed dataset
     - Tokenized datasets with correct format
     - A trained model with evaluation metrics
     - Validation accuracy > 85%
     - Test set predictions with confidence scores
     - At least **5 test predictions** on sample reviews

2. **Achieve reasonable performance**
   - Your model should achieve:
     - Training loss decreasing over epochs
     - Validation accuracy ≥ 85%
     - F1 score ≥ 0.85
   - If your scores are reasonable, or around this, don't stress out too much!
   - Include the final evaluation metrics in your notebook

3. **Test your classifier**
   - Run predictions on at least 5 different reviews
   - Include both positive and negative examples
   - Show the predicted sentiment and confidence score
   - Example test reviews are provided in the notebook

4. **Save your model**
   - Successfully save the fine-tuned model
   - Verify you can reload it for inference

5. **Push your work to your GitHub Classroom repo**
   - If working in Colab:
     - **File → Save a copy in GitHub** → choose your **Classroom repo** → commit message like "Completed Fine-Tuning project"
     - Or download the `.ipynb` and use Git locally:
       ```bash
       git clone <your-classroom-repo-url>
       cd <repo>
       git add MLB_Project_FineTuning_Template.ipynb
       git commit -m "Completed Fine-Tuning project"
       git push origin main
       ```
   - Or upload the file directly to the GitHub repo
   - Verify your notebook is visible in your repo on GitHub

6. **Submit on Canvas**
   - Post your **GitHub username** (e.g., `octocat`)
   - Post your **commit SHA** (a long hex string like `3f5c2a9...`)
     - How to find it: open your repo on GitHub → **Commits** → copy the **full SHA** of your final submission commit

> Your Canvas submission must include **both** your GitHub username and the **exact commit SHA** to get credit.

---

## 8) Tips & common pitfalls

### General tips
- **Use GPU**: Enable GPU in Colab settings (Runtime → Change runtime type)
- **Run incrementally**: Run each cell as you complete it to catch errors early
- **Monitor training**: Watch the loss decrease and accuracy increase during training
- **Save frequently**: Colab sessions timeout after 90 minutes of inactivity
- **Read the hints**: Every TODO has hints showing exactly what to do

### Common issues and solutions

**"CUDA out of memory" error:**
- Reduce batch size from 16 to 8 in the configuration cell
- Restart runtime: **Runtime → Restart runtime**
- Make sure no other notebooks are using GPU

**"No GPU available" / Training very slow:**
- Check Runtime type: **Runtime → Change runtime type → T4 GPU**
- If GPU quota exceeded, wait a few hours and try again
- Consider using Kaggle Notebooks (also offers free GPU)

**"Dataset loading timeout":**
- Your internet may be slow
- Try running just the dataset loading cell multiple times
- The dataset is cached after first successful load

**Model predictions all the same class:**
- Check that labels are balanced in your dataset
- Verify the model trained (loss should decrease)
- Try training for more epochs

**Training loss not decreasing:**
- Check learning rate isn't too high or too low
- Verify data is being loaded correctly
- Make sure you're using the right loss function

**"RuntimeError: Expected all tensors to be on the same device":**
- This is fixed in the answer key with device detection
- Make sure inputs are moved to the same device as the model
- See the `predict_sentiment` function for the fix

### Colab-specific tips
- **Save checkpoints**: The notebook saves after each epoch automatically
- **Use TensorBoard**: View training progress with `%load_ext tensorboard`
- **Check GPU usage**: Click the RAM/Disk indicator to see GPU utilization
- **Free tier limits**: ~12 hours GPU per day, sessions timeout after 90 minutes idle

---

## 9) Group Work

You are free to work in groups of 2-3 people! Just please submit a copy of your code anyway, it helps us keep track of who submitted what work.

When working in groups:
- Each person should run and test the complete notebook
- Make sure everyone understands the training process
- You can divide tasks (e.g., one person experiments with hyperparameters)
- But everyone should submit their own working notebook with results
- Discuss the evaluation metrics as a team

---

## 10) Understanding the code structure

To help you complete the TODOs, here's how the pieces fit together:

```
1. Load dataset
   └─> Amazon product reviews (question + context)

2. Data preprocessing
   └─> Combine title + content → rename columns → split into train/val/test

3. Tokenization
   └─> Load tokenizer → tokenize all text → create input_ids and attention_mask

4. Model setup
   └─> Load DistilBERT → add classification head → configure for 2 labels

5. Training configuration
   └─> Set hyperparameters → define metrics → create Trainer

6. Fine-tuning
   └─> Train for 3 epochs → evaluate after each epoch → save best model

7. Evaluation
   └─> Test on validation set → test on held-out test set → compute metrics

8. Inference
   └─> Load model → tokenize input → predict sentiment → calculate confidence
```

Complete the TODOs in order - each step builds on the previous one!

---

## 11) What you'll learn

By completing this project, you'll gain hands-on experience with:

✅ **Transfer learning** - Leveraging pre-trained models for new tasks  
✅ **Transformer models** - Understanding BERT-style architectures  
✅ **Tokenization** - Converting text to model inputs  
✅ **Training loops** - How models learn from data  
✅ **Hyperparameter tuning** - Learning rate, batch size, epochs  
✅ **Evaluation metrics** - Accuracy, precision, recall, F1  
✅ **Hugging Face ecosystem** - Transformers, Datasets, Trainer API  
✅ **Model deployment** - Saving and loading for inference  

These are core skills for any NLP practitioner and are used in production systems at every AI company!

---


## 12) Resources for learning more

Want to dive deeper into fine-tuning?

- [Hugging Face Course](https://huggingface.co/course/) - Free comprehensive NLP course
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training) - Official documentation
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) - Understanding the model
- [Transfer Learning in NLP](https://ruder.io/transfer-learning/) - Deep dive blog post
- [BERT Explained](http://jalammar.github.io/illustrated-bert/) - Visual guide to BERT

---

## 13) Need help?

- Re-read the comments and hints in the notebook cells
- Check the training logs for error messages
- Review the concepts in Sections 2 and 10 of this README
- Compare your code to the answer key (but try solving it first!)
- Ask questions in office hours or on the course forum

**Common debugging steps:**
1. Check GPU is enabled and being used
2. Verify the dataset loaded correctly (print a few examples)
3. Ensure tokenization produces correct format
4. Watch training loss - it should decrease
5. Compare your metrics to expected ranges

**Performance benchmarks:**
- After epoch 1: ~88-90% validation accuracy
- After epoch 2: ~91-93% validation accuracy  
- After epoch 3: ~92-94% validation accuracy
- Final test accuracy: ~90-93%

---

## 14) FAQ

**Q: Do I need a Hugging Face account?**  
A: No! The code works without an account. An account is only needed if you want to push your model to the Hub.

**Q: Why is my first epoch so slow?**  
A: The first epoch includes model compilation and optimization. Subsequent epochs are much faster.

**Q: How long should this project take?**  
A: Plan for 2-3 hours total:
- 30 min: Setup and understanding concepts
- 1 hour: Completing the TODOs
- 30 min: Training the model
- 30 min: Testing and evaluation
- 30 min: Submission

**Q: Can I use a different dataset?**  
A: Yes! But make sure it's in the same format (text + binary label). You'll need to adjust the data loading code.

**Q: My accuracy is stuck at 50%. What's wrong?**  
A: The model might not be learning. Check:
- Learning rate isn't too high (try 2e-5)
- Labels are correct (0 and 1, not other values)
- Data was tokenized properly
- Model is actually training (watch the loss)

**Q: Can I use ChatGPT/Claude to help with the code?**  
A: Yes, but make sure you understand what the code does! You'll need to explain your project and answer questions about how it works.

**Q: What if my model doesn't reach 85% accuracy?**  
A: Try:
- Training for more epochs (4-5)
- Using more data (increase from 5000 samples)
- Adjusting learning rate
- Using a larger model (bert-base instead of distilbert)

**Q: How do I know if I'm using the GPU?**  
A: Check the output after loading the model - it should say "CUDA available: True". Also, GPU usage will spike during training.

---

## 15) Understanding the metrics

Your model is evaluated using several metrics:

**Accuracy**: Percentage of correct predictions
- Good: ≥ 90%
- Acceptable: 70-90%
- Needs work: < 70%

**Precision**: Of all positive predictions, how many were actually positive?
- Important when false positives are costly

**Recall**: Of all actual positives, how many did we predict correctly?
- Important when false negatives are costly

**F1 Score**: Harmonic mean of precision and recall
- Balances both metrics
- Best single metric for classification

For sentiment analysis:
- **High precision**: Don't misclassify negative reviews as positive
- **High recall**: Don't miss actual positive reviews
- **High F1**: Good overall performance

---

## 16) Comparing to baseline

Your fine-tuned model should significantly outperform:
- **Random guessing**: 50% accuracy
- **Always predicting majority class**: ~50% accuracy (balanced dataset)
- **Pre-trained model without fine-tuning**: ~60-70% accuracy

Expected improvement from fine-tuning: **+20-30%** over pre-trained baseline!

This demonstrates the power of transfer learning and domain adaptation.

---

Good luck with your fine-tuning journey! 🚀

Remember: Fine-tuning is one of the most practical and widely-used techniques in modern NLP. You're learning skills that power everything from chatbots to content moderation systems. The ability to adapt pre-trained models to specific tasks is invaluable in industry!

---

**Pro tip**: After completing this project, you'll understand the foundations of how ChatGPT, Claude, and other AI assistants are trained. They use similar fine-tuning techniques (plus RLHF) on much larger models!
