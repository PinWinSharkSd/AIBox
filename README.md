#AIBox
is very simple LLM text generator with pytorch in python
build on FNet's neural network's
how to use?:
import and create new ai class and neural network:
```python
import aibox
ai = aibox()
ai.custom_model(vocab_size=100, d_model=100)
```
now you need to text data!, after getting your data init data to model:
```python
ai.init_data(YOUR_DATA)
```
after this you will train the model :
```python
ai.fit(
    epochs=100,
    lr=0.001,
    target=0.1, #stop reach
)
```
now you ready to generate text data from model with generate_text
```python
text = ai.generate_text(
"artificial intelligence is",
max_words=100,
end=['.'] #stop reach
)
print(text)
```
please wain the new feature:
```python
# Top-k for make new text
```
