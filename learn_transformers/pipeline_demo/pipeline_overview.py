# -*- coding: utf-8 -*-
"""
@File  : pipeline_overview.py
@Author: Yulong He
@Date  : 2023-03-22 10:33 a.m.
@Desc  : 
"""
from transformers import pipeline
import time


def sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    t0 = time.time()
    result = classifier("I've been waiting for a HuggingFace course my whole life.")
    print(result)
    result = classifier(
        ["I've been waiting for a HuggingFace course my whole life.",
         "I hate this so much!"]
    )
    print(result)
    print(f'elapsed time: {time.time() - t0}')


def zero_shot_classification():
    classifier = pipeline("zero-shot-classification")
    t0 = time.time()
    result = classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
    print(result)
    result = classifier(
        "I am a happy bear running in the grass, and I want to play around.",
        candidate_labels=["story", "education", "politics", "business"],
    )
    print(result)
    print(f'elapsed time: {time.time() - t0}')


def text_generation():
    generator = pipeline("text-generation")
    t0 = time.time()
    result = generator("In this course, we will teach you how to")
    print(result)
    print(f'elapsed time: {time.time() - t0}')

    generator = pipeline("text-generation", model="distilgpt2")
    t0 = time.time()
    result = generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )
    print(result)
    print(f'elapsed time: {time.time() - t0}')


def mask_filling():
    unmasker = pipeline("fill-mask")
    t0 = time.time()
    result = unmasker("This course will teach you all about <mask> models.", top_k=2)
    print(result)
    print(f'elapsed time: {time.time() - t0}')


def named_entity_recognition():
    ner = pipeline("ner", grouped_entities=True)
    t0 = time.time()
    result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    print(result)
    print(f'elapsed time: {time.time() - t0}')


def question_answering():
    question_answerer = pipeline("question-answering")
    t0 = time.time()
    result = question_answerer(
        question="Where do I work?",
        context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )
    print(result)
    print(f'elapsed time: {time.time() - t0}')


def summarization():
    summarizer = pipeline("summarization")
    t0 = time.time()
    result = summarizer(
        """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.

        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
    """
    )
    print(result)
    print(f'elapsed time: {time.time() - t0}')


def translation():
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
    t0 = time.time()
    result = translator("你好，我的名字叫何雨龙。我是一名软件研发工程师。")
    print(result)
    print(f'elapsed time: {time.time() - t0}')


def feature_extraction():
    extractor = pipeline("feature-extraction", model="bert-base-uncased")
    t0 = time.time()
    result = extractor("This is a simple test.", return_tensors=True)
    print(result.shape)
    print(f'elapsed time: {time.time() - t0}')


if __name__ == '__main__':
    feature_extraction()
