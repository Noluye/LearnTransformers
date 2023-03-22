from transformers import pipeline
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    classifier = pipeline("sentiment-analysis")
    t0 = time.time()
    result = classifier("I've been waiting for a HuggingFace course my whole life.")
    print(f'elapsed time: {time.time() - t0}')
    print(result)

